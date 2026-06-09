"""
生成样本合规公告图片（模拟扫描件/截图场景）
用法: conda run -n 2024_VLM python practice_vlm/generate_images.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

ANNOUNCEMENTS = {
    "p0_violation.png": {
        "title": "关于购买资产的公告",
        "body": """一、交易概述

本公司拟以现金方式购买恒盛投资有限公司持有的
恒盛数据科技有限公司100%股权，交易价格为人民币
8,500万元，支付方式为现金支付，分两期付款。
本次交易不构成重大资产重组。

二、交易对方信息

交易对方：恒盛投资有限公司
注册地：广东省深圳市南山区科技园南区
法定代表人：王建国
注册资本：人民币5,000万元

三、其他说明

本公司控股股东为王建国先生，直接持有公司38%股份。
本次交易无需提交股东大会审议。""",
        "violation": "P0：关联交易未认定（交易对方法定代表人即为控股股东）",
    },
    "p1_violation.png": {
        "title": "关于购买资产的公告",
        "body": """一、交易概述

本公司拟以现金方式向自然人李明购买其持有的
明远物流有限公司60%股权，交易价格为人民币
3,200万元，一次性现金支付。本次交易不构成
重大资产重组，不构成关联交易。

二、交易对方信息

交易对方：李明（自然人）
与上市公司关系：无关联关系

三、董事会意见

本次交易经董事会审议通过，独立董事表示认可。""",
        "violation": "P1：自然人信息缺失（国籍/住所/身份证件/犯罪记录/资金来源均未披露）",
    },
    "compliant.png": {
        "title": "关于购买资产的公告",
        "body": """一、交易概述

本公司拟购买博晟创投有限公司持有的云启软件科技
有限公司51%股权，交易价格人民币1.2亿元，现金
支付，分三期。本次交易不构成重大资产重组，
不构成关联交易，无需提交股东大会审议。

二、交易对方信息

交易对方：博晟创投有限公司
注册地：北京市朝阳区建国路88号
法定代表人：陈浩    注册资本：人民币1亿元
与上市公司关系：无关联关系

三、董事会及独立董事意见

经全体董事审议通过，独立董事发表事前认可意见：
本次交易定价公允，不损害上市公司及中小股东利益。""",
        "violation": None,
    },
}


def render_announcement(title: str, body: str, violation: str | None, output_path: Path):
    W, H = 800, 900
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 尝试加载系统中文字体，找不到就用默认
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_title = font_body = ImageFont.load_default()
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font_title = ImageFont.truetype(fp, 22)
                font_body = ImageFont.truetype(fp, 16)
                break
            except Exception:
                continue

    # 标题
    draw.rectangle([0, 0, W, 50], fill=(220, 230, 245))
    draw.text((W // 2, 25), title, font=font_title, fill=(20, 40, 80), anchor="mm")

    # 正文
    y = 70
    for line in body.strip().split("\n"):
        draw.text((40, y), line, font=font_body, fill=(30, 30, 30))
        y += 24

    # 违规标注
    if violation:
        draw.rectangle([20, H - 60, W - 20, H - 10], fill=(255, 240, 240), outline=(200, 50, 50), width=2)
        draw.text((30, H - 45), f"[审核标注] {violation}", font=font_body, fill=(180, 30, 30))

    img.save(output_path)
    print(f"生成: {output_path.name}")


for filename, content in ANNOUNCEMENTS.items():
    render_announcement(
        title=content["title"],
        body=content["body"],
        violation=content["violation"],
        output_path=OUTPUT_DIR / filename,
    )

print(f"\n共生成 {len(ANNOUNCEMENTS)} 张图片 → {OUTPUT_DIR}")
