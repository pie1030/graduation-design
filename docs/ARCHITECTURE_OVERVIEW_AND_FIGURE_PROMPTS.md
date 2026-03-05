# DeltaVLM 模型框架说明与架构图生成指令

本文档描述当前 DeltaVLM 的 **CD（变化检测）**、**CC（变化描述）** 以及 **CC+CD 联合** 框架，并给出三条供 [PaperBanana](https://github.com/dwzhu-pku/PaperBanana) + Google Gemini 3.1 Flash (image preview) 生成顶刊风格 Architecture Overview 的指令。

---

## 一、当前模型框架总览

### 1. 共享底座

- **输入**：双时相图像 \(I_A\), \(I_B\)（如 224×224），经同一 **EVA-ViT-G** 视觉编码器提取 token 序列。
- **EVA-ViT**：冻结，输出形状为 \((B, 257, 1408)\)（CLS + 16×16 patch tokens）。

---

### 2. CD（Change Detection）分支

**目标**：输出像素级变化掩码 \(\hat{M} \in \{0,1,2\}^{H\times W}\)（背景 / 道路 / 建筑）。

**数据流**：

1. **Path 1（冻结）**  
   - \(I_A, I_B \rightarrow\) **EVA-ViT** \(\rightarrow\) \(\mathbf{f}_A^{\text{lr}}, \mathbf{f}_B^{\text{lr}}\) \((B, 257, 1408)\)  
   - **Semantic Adapter**：去 CLS、线性投影 + LayerNorm、reshape 并上采样/对齐到 14×14 → \(\mathbf{s}_A, \mathbf{s}_B\) \((B, 256, 14, 14)\)

2. **Path 2（可训练）**  
   - \(I_A, I_B \rightarrow\) **HR Encoder（ResNet-18，Siamese 权重共享）**  
   - 多尺度特征：\(\mathbf{h}_1 (56\times56)\), \(\mathbf{h}_2 (28\times28)\), \(\mathbf{h}_3 (14\times14)\)，通道分别为 64, 128, 256。

3. **语义注入与 CSRM**  
   - 在 14×14 尺度：\(\mathbf{s}_A \oplus \mathbf{h}_3^A\)、\(\mathbf{s}_B \oplus \mathbf{h}_3^B\) 经卷积融合后，送入 **Spatial CSRM**（门控 + 上下文），得到差异感知的深层特征。

4. **多尺度差异融合（Change-Agent 风格）**  
   - 在 56、28、14 三个尺度上：\(\text{diff} = \text{conv}_{\text{dif}}(\mathbf{h}_B - \mathbf{h}_A) + \cos(\mathbf{h}_A, \mathbf{h}_B)\)，再 \(\text{conv}_{\text{fuse}}([\mathbf{h}_A, \text{diff}, \mathbf{h}_B])\) 得到每尺度融合特征。

5. **FPN 解码器**  
   - 自顶向下：14→28→56→112，逐级上采样并与对应尺度融合特征相加，最后 **Seg Head**（Conv + 上采样）→ 3 类 logits，插值到 256×256 得到 \(\hat{M}\)。

**训练**：仅训练 HR Encoder、Semantic Adapter、CSRM、conv_dif/fuse、FPN、Seg Head；EVA-ViT 冻结。损失为带类别权重的 Cross-Entropy。

---

### 3. CC（Change Captioning）分支

**目标**：根据 \(I_A, I_B\) 和可选文本 prompt 生成自然语言变化描述 \(\hat{y}\)。

**数据流**：

1. **EVA-ViT**  
   - \(I_A, I_B \rightarrow\) \(\mathbf{f}_A, \mathbf{f}_B\) \((B, 257, 1408)\)。

2. **CSRM（仅推理 generate 路径）**  
   - \(\Delta = \mathbf{f}_B - \mathbf{f}_A\)  
   - 门控与上下文：\(\mathbf{c}_A = \tanh(W_c \Delta + W_c' \mathbf{f}_A)\)，\(g_A = \sigma(W_g \Delta + W_g' \mathbf{f}_A)\)，\(\tilde{\mathbf{f}}_A = g_A \odot \mathbf{c}_A\)；\(\tilde{\mathbf{f}}_B\) 对称。  
   - \(\mathbf{e}_A = \text{context3}([\mathbf{f}_A, \Delta, \tilde{\mathbf{f}}_A])\)，\(\mathbf{e}_B = \text{context3}([\mathbf{f}_B, \Delta, \tilde{\mathbf{f}}_B])\)。

3. **Q-Former**  
   - 将 \([\mathbf{e}_A, \mathbf{e}_B]\) 与 query tokens、文本 prompt 一起输入 Q-Former，得到 query 输出。

4. **LLM**  
   - query 输出经 **llm_proj** 与 prompt 的 embedding 拼接，输入 **Vicuna LLM**，自回归生成 \(\hat{y}\)。

**训练**：可冻结 EVA-ViT 与 CSRM，训练 Q-Former + LLM；或端到端微调部分模块。

---

### 4. CC + CD 联合

- **同一模型**：共享 **EVA-ViT** 与双时相输入 \(I_A, I_B\)。  
- **CD 分支**：在 EVA-ViT 之上接 HR Encoder、Semantic Adapter、CSRM（空间版）、多尺度融合、FPN、Seg Head → \(\hat{M}\)。  
- **CC 分支**：在 EVA-ViT 之上接 CSRM（序列版）、Q-Former、LLM → \(\hat{y}\)。  
- **联合使用**：前向时既可只跑 CD（如仅需掩码），只跑 CC（仅需描述），也可两者都跑（多任务评估或应用）。  
- **训练策略**：当前实现中，CD 训练时可删除 LLM 以省显存，仅保留并训练 CD 相关模块；CC 训练时使用上述 generate 路径的 CSRM + Q-Former + LLM。

---

## 二、绘图规范（顶刊风格）

- **配色**：严禁高饱和霓虹色。主色：科技柔和色，如淡蓝 `#E6F3FF`，辅色：柔和橙色（如 `#F5D4A0` 或 `#E8C9A0`）。可少量深灰/深蓝作边框或文字。  
- **背景**：纯白 `#FFFFFF` 或极淡灰 `#F8F9FA`，严禁纯黑背景。  
- **几何**：模块用圆角矩形；仅矩阵/张量形状用直角矩形表示。箭头清晰，数据流从左到右或自上而下。  
- **字体**：层级分明——数学符号与变量用 LaTeX 风格（如 \(I_A, \mathbf{f}_A, \hat{M}\)）；普通标签用 Sans-Serif（如 “EVA-ViT”, “Q-Former”, “Mask”）。

---

## 三、供 PaperBanana / Gemini 使用的三条指令

将下面三条指令分别作为「方法描述 + 图说」输入 PaperBanana（配合 Google Gemini 3.1 Flash image preview），可生成三张对应的 Architecture Overview 图。每条已包含上述绘图规范与所需元素。

---

### 指令 1：CD 分支架构图（Change Detection Only）

```
Draw a single, publication-quality architecture overview figure for a dual-path Change Detection (CD) model. The figure must show only the CD pipeline from dual-time images to the predicted change mask.

Content to include:
- Left: two input boxes (rounded rectangles), labeled "I_A" and "I_B" (use serif/LaTeX-style for I_A, I_B). Both feed into two parallel paths.
- Path 1 (top): a rounded rectangle "EVA-ViT (frozen)" outputting low-resolution semantic features "f_lr" (16×16, 1408d). Then a block "Semantic Adapter" that projects and reshapes to spatial 14×14, 256d.
- Path 2 (bottom): a rounded rectangle "HR Encoder (ResNet-18, trainable)" with shared weights for both images, outputting multi-scale features: "h_1 (56×56)", "h_2 (28×28)", "h_3 (14×14)".
- Middle: "Semantic injection" at 14×14: combine adapter output with h_3, then a block "Spatial CSRM" (gate + context). Then "Multi-scale diff fusion" at 56, 28, 14: conv_dif, cosine similarity, conv_fuse, with arrows from h_A and h_B and diff.
- Right: "FPN decoder" (top-down, 14→28→56→112) with skip additions, then "Seg Head" and final output "M̂" (3-class mask, 256×256). Use LaTeX-style for M̂ and dimension notations.

Style (strict):
- Colors: soft tech palette only. Primary fill #E6F3FF (light blue), secondary #F5D4A0 or #E8C9A0 (soft orange) for key modules. No neon or saturated colors.
- Background: pure white #FFFFFF or very light gray #F8F9FA. No black background.
- All main modules: rounded rectangles. Use right-angle rectangles only for small tensor shape annotations (e.g. "256×256").
- Typography: math symbols and variables (I_A, I_B, f_lr, M̂, dimensions) in LaTeX/serif style; all other labels (EVA-ViT, HR Encoder, CSRM, FPN, Seg Head) in clean Sans-Serif.
- Arrows: clear, same soft color for data flow. Keep the layout left-to-right or top-to-bottom for readability.
```

---

### 指令 2：CC 分支架构图（Change Captioning Only）

```
Draw a single, publication-quality architecture overview figure for a Change Captioning (CC) model. The figure must show only the CC pipeline from dual-time images to the generated text caption.

Content to include:
- Left: two input boxes (rounded rectangles), labeled "I_A" and "I_B" (LaTeX-style). Both feed into "EVA-ViT" producing "f_A" and "f_B" (257, 1408).
- Middle: a block "CSRM" (Cross-temporal Spatial Reasoning Module): compute difference "Δ = f_B − f_A", then gate and context: "c = tanh(W·Δ + W'·f)", "g = σ(...)", "f̃ = g ⊙ c" for both time steps. Then "context3" fuses [f, Δ, f̃] to "e_A" and "e_B". Use compact LaTeX-style for Δ, f̃, σ, ⊙.
- Next: concatenate "e_A" and "e_B" into a single sequence, then a rounded rectangle "Q-Former" with "query tokens" and optional "text prompt" as inputs. Output: "query output".
- Right: "llm_proj" (small block), then "Vicuna LLM" (large rounded rectangle) with "prompt embedding" concatenated with projected query. Output: "ŷ" (generated caption text). Use LaTeX-style for ŷ and all math.

Style (strict):
- Colors: soft tech palette. Primary #E6F3FF, secondary soft orange #F5D4A0 or #E8C9A0. No neon or saturated colors.
- Background: pure white #FFFFFF or very light gray #F8F9FA. No black background.
- Modules: rounded rectangles; right-angle only for small shape/text annotations.
- Typography: variables (f_A, f_B, Δ, f̃, ŷ, σ, ⊙) in LaTeX/serif; labels (EVA-ViT, CSRM, Q-Former, Vicuna LLM) in Sans-Serif.
- Arrows: clear, soft-tone data flow. Layout left-to-right.
```

---

### 指令 3：CC + CD 联合架构图（Unified Model）

```
Draw a single, publication-quality architecture overview figure for a unified model that supports both Change Detection (CD) and Change Captioning (CC) from the same dual-time images I_A and I_B.

Content to include:
- Top or center-left: one "EVA-ViT (frozen)" block taking I_A and I_B (two arrows from two small input boxes "I_A", "I_B"). Output: "f_A, f_B" (257, 1408). This is the shared backbone.
- From the shared backbone, split into two branches (draw a clear fork):
  - CD branch (e.g. lower or right-bottom): "Semantic Adapter" → spatial 14×14; parallel "HR Encoder (ResNet-18)" → multi-scale h_1, h_2, h_3; "Semantic injection + Spatial CSRM" at 14×14; "Multi-scale diff fusion" (56/28/14); "FPN decoder" → "Seg Head" → output "M̂" (change mask, 3-class).
  - CC branch (e.g. upper or right-top): "CSRM" (sequence-level gate + context on f_A, f_B) → "e_A, e_B"; "Q-Former" (query + image embeds); "llm_proj" → "Vicuna LLM" → output "ŷ" (caption).
- Add a short label near the fork: "Shared visual features" or "Dual-time tokens".
- Use LaTeX-style for I_A, I_B, f_A, f_B, M̂, ŷ and dimension notations; Sans-Serif for all module names.

Style (strict):
- Colors: soft tech palette. Primary #E6F3FF, secondary #F5D4A0 or #E8C9A0. No neon or saturated colors.
- Background: pure white #FFFFFF or very light gray #F8F9FA. No black background.
- All main modules: rounded rectangles; right-angle only for tensor/shape annotations.
- Typography: math in LaTeX/serif; labels in Sans-Serif. Arrows clear and soft-tone. Layout readable (e.g. shared backbone left, two branches diverge to right or bottom).
```

---

## 四、使用方式建议

1. 在 PaperBanana 的 **Generate Candidates** 中，将上述某一条指令粘贴到「方法描述」或「图说」输入框。  
2. 在配置中选用 **Google Gemini 3.1 Flash (image preview)** 作为生成模型。  
3. 若支持，在「风格」或「补充说明」中再次强调：**No neon colors; background white or #F8F9FA; rounded rectangles; LaTeX for math, Sans-Serif for labels.**  
4. 生成多张候选后，可选用最符合顶刊排版与可读性的一张，或据此微调文案再生成一轮。

引用 PaperBanana 时可按其仓库说明引用：  
[dwzhu-pku/PaperBanana](https://github.com/dwzhu-pku/PaperBanana) — PaperBanana: Automating Academic Illustration For AI Scientists.
