# TakeApart 需求文档

> 版本: 2.0 | 日期: 2026-07-03 | 状态: Draft

---

## 1. detect_elements — 轻量元素检测

### 1.1 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| image_path | string | ✅ | — | 图片文件路径 |
| output_path | string | ❌ | "" | 标注可视化输出路径（空=不输出） |
| min_area_ratio | float | ❌ | 0.002 | 最小面积比例 |
| blur_kernel | int | ❌ | 5 | 高斯模糊核大小（奇数，0=关闭） |
| canny_low | int | ❌ | 50 | Canny 低阈值 |
| canny_high | int | ❌ | 150 | Canny 高阈值 |

### 1.2 输出格式

```json
{
  "image_info": {"path": "...", "width": 1920, "height": 1080},
  "elements": [
    {
      "id": 0,
      "bbox": {"x": 100, "y": 200, "width": 300, "height": 150},
      "area_ratio": 0.02,
      "area_pixels": 32400,
      "contour_points": [[100,200],[400,200],[400,350],[100,350]],
      "aspect_ratio": 2.0
    }
  ],
  "total_count": 5,
  "visualization_path": "output/image_detected.png"
}
```

### 1.3 算法流程

```
原图 → 灰度化 → 高斯模糊 → Canny 边缘检测
  → 膨胀(闭合小缝) → findContours
  → 过滤面积/长宽比 → 按面积排序 → 输出
```

---

## 2. segment_element — 精确分割

### 2.1 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| image_path | string | ✅ | — | 图片文件路径 |
| boxes | array | ✅ | — | 边界框列表 |
| output_mode | string | ❌ | "png" | mask / png / both |
| output_dir | string | ❌ | "" | PNG 输出目录 |
| padding | int | ❌ | 10 | 裁剪边距 |
| crop | bool | ❌ | true | 是否裁剪到元素边界 |
| edge_refine | bool | ❌ | true | 是否启用 Canny 边缘精修 |

### 2.2 boxes 数组格式

```json
[
  {"x1": 100, "y1": 200, "x2": 400, "y2": 350, "label": "标题"},
  {"x1": 500, "y1": 100, "x2": 800, "y2": 400, "label": "图标"}
]
```

### 2.3 输出格式

```json
{
  "image_info": {"path": "...", "width": 1920, "height": 1080},
  "results": [
    {
      "index": 0,
      "label": "标题",
      "input_box": {"x1": 100, "y1": 200, "x2": 400, "y2": 350},
      "bbox": {"x": 105, "y": 205, "width": 290, "height": 140},
      "area_ratio": 0.019,
      "confidence": 0.92,
      "mask_base64": "iVBORw0KGgo...",
      "output_path": "output/image_00_标题.png",
      "extracted_size": {"width": 310, "height": 160}
    }
  ],
  "total_count": 2
}
```

### 2.4 算法流程

```
框选坐标 → SAM 粗分(mask + logits)
  → logit 阈值化 → 初始二值 mask
  → Canny 边缘检测(ROI 内)
  → 距离变换 → 边缘吸附(mask 边界→最近 Canny 边缘)
  → 形态学闭运算(填缝) + 洪泛填充(去孔洞)
  → 轮廓重建 → 硬边缘 mask
  → 输出 mask / PNG / both
```
