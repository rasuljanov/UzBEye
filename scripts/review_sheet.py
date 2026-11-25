"""Render the selected reference boxes and IDs for a bilingual reviewer."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from src.common import read_json
from src.data.dataset import open_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--form", default="review.json")
    args = parser.parse_args()
    rows = read_json(args.root / args.form)
    for page in range(0, len(rows), 16):
        group = rows[page:page+16]
        sheet = Image.new("RGB", (960, ((len(group)+3)//4)*206), "white")
        for i, row in enumerate(group):
            image = open_image(args.root, row)
            x,y,w,h = row["bbox"]
            ImageDraw.Draw(image).rectangle((x,y,x+w,y+h), outline="lime", width=4)
            image.thumbnail((234,180))
            left, top = (i%4)*240, (i//4)*206
            sheet.paste(image, (left,top))
            ImageDraw.Draw(sheet).text((left+3,top+182), f"{row['split']}   {row['id']}", fill="black")
        path = args.root / f"{Path(args.form).stem}-{page//16+1}.jpg"
        sheet.save(path)
        print(path)


if __name__ == "__main__":
    main()
