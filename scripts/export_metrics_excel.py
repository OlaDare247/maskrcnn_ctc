import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-xlsx", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_xlsx = Path(args.out_xlsx)
    data = json.loads(metrics_path.read_text())

    summary = pd.DataFrame([
        {"Metric": "Mean Dice / DCE", "Value": data.get("mean_dice")},
        {"Metric": "Mean Jaccard / JCD", "Value": data.get("mean_jaccard")},
        {"Metric": "Number of Images", "Value": data.get("num_images")},
    ])

    frames = pd.DataFrame(data.get("per_image", []))

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        frames.to_excel(writer, sheet_name="Per_Frame_Metrics", index=False)

        workbook = writer.book
        fmt_header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        fmt_num = workbook.add_format({"num_format": "0.0000"})
        fmt_title = workbook.add_format({"bold": True, "font_size": 14})

        ws = writer.sheets["Summary"]
        ws.write("A1", "Metric", fmt_header)
        ws.write("B1", "Value", fmt_header)
        ws.set_column("A:A", 25)
        ws.set_column("B:B", 18, fmt_num)

        ws2 = writer.sheets["Per_Frame_Metrics"]
        for col, name in enumerate(frames.columns):
            ws2.write(0, col, name, fmt_header)
        ws2.set_column("A:C", 22)
        ws2.set_column("D:H", 18, fmt_num)

    print(f"Saved Excel report to: {out_xlsx}")

if __name__ == "__main__":
    main()