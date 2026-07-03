# TakeApart 技术规格文档 (SPEC)

> 版本: 2.0 | 日期: 2026-07-03 | 状态: Draft

---

## 1. 系统架构

```
MCP Client (Claude Desktop / Claude Code)
    │  stdio (JSON-RPC 2.0)
    ▼
┌─────────────────────────────────────────────┐
│         TakeApart MCP Server                │
│                                             │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │ detect_elements  │  │ segment_element  │  │
│  │ Canny + 轮廓     │  │ SAM + Canny精修  │  │
│  │ 无 GPU          │  │ 需要 GPU         │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           │                    │             │
│  ┌────────┴────────┐  ┌───────┴──────────┐  │
│  │ utils/           │  │ models/          │  │
│  │ image_io.py      │  │ sam_model.py     │  │
│  │ mask_ops.py      │  │ (SAM 2 单例)     │  │
│  │ edge_refine.py   │  └──────────────────┘  │
│  └─────────────────┘                         │
└─────────────────────────────────────────────┘
```

---

## 2. 工具规格

### 2.1 detect_elements

**算法**：纯传统 CV，零 GPU 依赖

```python
灰度化 → GaussianBlur(kernel) → Canny(low, high)
  → dilate(3×3, iter=1) → findContours(RETR_EXTERNAL)
  → filterByArea(min_area_ratio) → sortByArea(desc)
```

**参数**：image_path, output_path, min_area_ratio, blur_kernel, canny_low, canny_high

### 2.2 segment_element

**算法**：SAM 粗分 + Canny 边缘精修

```python
# SAM 粗分
box → SAM.predict → masks, scores, logits

# 边缘精修
prob = sigmoid(logits)
binary = (prob > 0.5)
edges = Canny(roi)
dist = distanceTransform(edges)
snapped = snapEdgeToCanny(binary, dist, radius=5)
closed = morphologyEx(snapped, MORPH_CLOSE)
filled = floodFill(closed)
result = morphologyEx(filled, MORPH_OPEN)
```

**参数**：image_path, boxes[], output_mode, output_dir, padding, crop, edge_refine

---

## 3. 文件结构

```
src/takeapart/
├── server.py           # MCP Server 入口 (2 个工具)
├── tools/
│   ├── detect.py       # Canny + 轮廓检测
│   └── segment.py      # SAM + Canny 边缘精修
├── models/
│   └── sam_model.py    # SAM 2 懒加载单例
└── utils/
    ├── image_io.py     # 图像 I/O
    ├── mask_ops.py     # mask 操作
    └── edge_refine.py  # Canny 边缘精修算法
```

---

## 4. MCP 工具列表

| 工具名 | 输入 | 输出 |
|--------|------|------|
| detect_elements | image_path + 可选参数 | 元素列表 + 可选可视化图 |
| segment_element | image_path + boxes[] | mask / PNG / both |

---

## 5. 部署

### 5.1 环境要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10 | 3.12 |
| GPU | 不需要(detect) / NVIDIA 6GB(segment) | NVIDIA 8GB |
| RAM | 4 GB | 16 GB |

### 5.2 MCP 配置

```json
{
  "mcpServers": {
    "takeapart": {
      "command": "python",
      "args": ["-m", "takeapart.server"],
      "cwd": "D:\\Application\\TakeApart"
    }
  }
}
```

---

## 6. 日志

- 输出：stderr
- 格式：`{asctime} [{name}] {levelname}: {message}`
- 关键日志：模型加载、工具调用、异常
