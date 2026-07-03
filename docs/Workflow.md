# TakeApart 工作流定义

> 版本: 1.0 | 日期: 2026-07-03

---

## 核心原则

**CV 测量，AI 理解。** 各取所长，互补短板。

| 维度 | 数据来源 | 理由 |
|------|---------|------|
| 位置 (x, y) | CV detect | 像素精确 |
| 尺寸 (w, h) | CV detect | 像素精确 |
| 字体大小 | CV detect (font_size_px) | 像素精确 |
| 元素类型 | CV 初判 → 视觉 AI 修正 | CV 粗分类，AI 语义修正 |
| 文字内容 | 视觉 AI | CV 不读内容 |
| 字体样式 | 视觉 AI | CV 不识别字体 |
| 颜色 | 视觉 AI | CV 可做但 AI 更准 |
| 语义关系 | 视觉 AI | CV 不理解语义 |

---

## 标准工作流

### Phase 1: CV 扫描（detect_elements）
```
输入: 图片路径
输出: 元素列表（id, type, bbox, font_size_px, aspect_ratio, area_ratio）
```
- 零 GPU，快速 (< 1s)
- 提供精确的位置/尺寸/类型
- 输出可视化标注图（可选）

### Phase 2: 视觉 AI 解读
```
输入: 原图 + detect 结果（作为上下文）
输出: 每个元素的内容/语义属性
```
- 视觉 AI 看图 + 参考 detect 的坐标
- 逐元素读取内容、识别字体、判断颜色
- 修正 CV 的类型判断（如有误判）
- 不修改位置/尺寸（以 CV 为准）

### Phase 3: 精确分割（segment_element，按需）
```
输入: 需要抠图的元素坐标（来自 Phase 1 的 bbox）
输出: 透明 PNG / mask
```
- 仅在需要导出元素文件时调用
- SAM + Canny 边缘精修
- 硬边缘，PS 级别质量

---

## 冲突解决规则

| 冲突类型 | 优先听从 | 原因 |
|---------|---------|------|
| 位置坐标 | CV | 像素精确 vs 估算 |
| 尺寸大小 | CV | 像素精确 vs 估算 |
| 元素边界 | CV | 轮廓精确 |
| 元素类型 | 视觉 AI | 语义理解更强 |
| 文字内容 | 视觉 AI | CV 不读内容 |
| 字体/颜色 | 视觉 AI | CV 识别能力有限 |
| 对齐方式 | 视觉 AI | left/center/right |
| 行间距 | 视觉 AI | 倍数，如 1.0/1.5/2.0 |
| 段后间距 | 视觉 AI | 像素值 |
| 字重(粗体) | 视觉 AI | bold/normal |

---

## 视觉 AI 输入模板

Phase 1 完成后，将以下 JSON 作为上下文传给视觉 AI：

```json
{
  "任务": "请根据图片和以下元素检测结果，逐个补充每个元素的内容描述",
  "规则": {
    "位置和尺寸": "以 detect 结果为准，不要修改",
    "元素类型": "可以修正 CV 的判断",
    "需要补充": "文字内容、字体样式、颜色、语义描述"
  },
  "image_size": {"width": 1098, "height": 504},
  "detected_elements": [
    {
      "id": 0,
      "type": "text",
      "bbox": {"x": 726, "y": 359, "width": 269, "height": 64},
      "font_size_px": 64,
      "area_ratio": 0.028
    }
  ]
}
```

---

## 最终输出结构

Phase 1 + Phase 2 合并后的完整元素属性：

```json
{
  "id": 0,
  "type": "text",
  "bbox": {"x": 726, "y": 359, "width": 269, "height": 64},
  "font_size_px": 64,
  "content": "这是视觉AI读取的文字内容",
  "font_family": "宋体",
  "color": "#333333",
  "bold": false,
  "align": "left",
  "line_spacing": 1.5,
  "paragraph_spacing": 0,
  "description": "视觉AI的语义描述"
}
```
