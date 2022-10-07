import os
import shutil


def create_html(input_directory, output_dir, output_folder_name, screen_w=1600):
    if not os.path.exists(os.path.join(output_dir, output_folder_name)):
        os.makedirs(os.path.join(output_dir, output_folder_name))
        os.makedirs(os.path.join(output_dir, output_folder_name, "gifs"))

    folders = os.listdir(input_directory)
    gif_paths = []
    for folder in folders:
        path = os.path.join(input_directory, folder, f"{folder}_semantic.gif")
        dest_path = os.path.join(
            output_dir, output_folder_name, "gifs", f"{folder}_semantic.gif"
        )
        if os.path.exists(path):
            shutil.copy(path, dest_path)
            gif_paths.append(os.path.join("gifs", f"{folder}_semantic.gif"))

        path = os.path.join(input_directory, folder, f"{folder}_instance.gif")
        dest_path = os.path.join(
            output_dir, output_folder_name, "gifs", f"{folder}_instance.gif"
        )
        if os.path.exists(path):
            shutil.copy(path, dest_path)

    rows_texts = ""
    for gif_p in gif_paths:
        rows_texts += "<tr>\n"
        rows_texts += f'\t<td><img src="{gif_p}" width="{screen_w/2}"/></td>\n'
        rows_texts += f'\t<td><img src="{gif_p.replace("semantic","instance")}" width="{screen_w/2}"/></td>\n'
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
