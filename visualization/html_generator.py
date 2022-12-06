import os
import shutil


def create_html(input_directory, output_dir, output_folder_name, screen_w=1600):
    if not os.path.exists(os.path.join(output_dir, output_folder_name)):
        os.makedirs(os.path.join(output_dir, output_folder_name))
        os.makedirs(os.path.join(output_dir, output_folder_name, "pngs"))

    folders = os.listdir(input_directory)
    png_paths = []
    for folder in folders:
        path = os.path.join(input_directory, folder, f"{folder}_semantic.png")
        dest_path = os.path.join(
            output_dir, output_folder_name, "pngs", f"{folder}_semantic.png"
        )
        if os.path.exists(path):
            shutil.copy(path, dest_path)
            png_paths.append(os.path.join("pngs", f"{folder}_semantic.png"))

        path = os.path.join(input_directory, folder, f"{folder}_instance.png")
        dest_path = os.path.join(
            output_dir, output_folder_name, "pngs", f"{folder}_instance.png"
        )
        if os.path.exists(path):
            shutil.copy(path, dest_path)

    rows_texts = ""
    for png_p in png_paths:
        rows_texts += "<tr>\n"
        rows_texts += f'\t<td><img src="{png_p}" width="{screen_w/2}"/></td>\n'
        rows_texts += f'\t<td><img src="{png_p.replace("semantic","instance")}" width="{screen_w/2}"/></td>\n'
        rows_texts += "</tr>\n"

    html_text = f"""
    <html>
        <table>
            {rows_texts}
        </table> 
    </html>
    """

    with open(
        os.path.join(output_dir, output_folder_name, f"visualizer.html"), "w"
    ) as f:
        f.write(html_text)


# create_html(
#     "/space/ariyanzarei/sorghum_segmentation/results/2020-08-06",
#     "/space/ariyanzarei/sorghum_segmentation/results/",
#     "test_html",
#     col=2,
# )
