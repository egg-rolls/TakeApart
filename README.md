# TakeApart 🔍

**AI 视觉拆解工具** — 让 AI 拥有 PS 级别的眼睛和手

TakeApart 是一个 MCP 工具服务，让 AI 能够对图片进行像素级元素拆解。

## 核心能力

| 工具 | 内核 | GPU | 功能 |
|------|------|-----|------|
| `detect_elements` | Canny + 轮廓 | ❌ | 快速发现图中大致元素区域 |
| `segment_element` | SAM + Canny 精修 | ✅ | PS 式硬边缘精确分割，支持批量 |

## 适用场景

- 🎨 **设计稿** — 拆分 UI 组件、提取图标
- 📄 **海报** — 分离标题、图片、装饰元素
- 📊 **PPT** — 提取图表、文字、Logo
- 🖼️ **人像/物体** — 像素级精确抠图

## 安装

```bash
# 1. 安装项目依赖
pip install -e .

# 2. 安装 SAM 2（从 GitHub 源码，segment 工具需要）
pip install git+https://github.com/facebookresearch/sam2.git

# 3. 下载模型权重（~300MB）
# https://github.com/facebookresearch/sam2/releases
# 放入 models/ 目录
```

## MCP 配置

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

## 工具使用

### detect_elements — 快速发现元素
```json
{"image_path": "poster.png", "output_path": "output/detected.png"}
```

### segment_element — 精确分割
```json
{
  "image_path": "poster.png",
  "boxes": [
    {"x1": 100, "y1": 200, "x2": 400, "y2": 350, "label": "标题"},
    {"x1": 500, "y1": 100, "x2": 800, "y2": 400, "label": "图标"}
  ],
  "output_mode": "png",
  "output_dir": "output/elements"
}
```

## License

MIT
