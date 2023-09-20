import os
from zipfile import ZipFile
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile, rmtree
import tempfile
import traceback

RGB_ZIP_FP = Path("datasets/tufts-face-database-computerized-sketches-td-cs.zip")
IR_ZIP_FP = Path("datasets/tufts-face-database-thermal-td-ir.zip")
TARGET_DIR = Path("datasets/RGB_Thermal_dataset")

TMP_DIR = Path(tempfile.mkdtemp("tmp"))

if not os.path.exists(TARGET_DIR):
    os.mkdir(str(TARGET_DIR))
else:
    rmtree(TARGET_DIR)
    os.mkdir(str(TARGET_DIR))
    Path(TARGET_DIR / ".gitkeep").touch()

try:
    with ZipFile(RGB_ZIP_FP, 'r') as fzip:
        fzip.extractall(TMP_DIR / RGB_ZIP_FP.stem)

    with ZipFile(IR_ZIP_FP, 'r') as fzip:
        fzip.extractall(TMP_DIR / IR_ZIP_FP.stem)

    # Containing tuples as (source, dest)
    list_to_copy = []
    for d_path, d_names, f_names in os.walk(TMP_DIR):
        if len(f_names) == 0:
            pass
        else:
            for file in f_names:
                if not any(["TD_IR_A" in x or "TD_CS" in x for x in Path(d_path).parts]):
                    # print(d_path)
                    pass
                elif ("RGB" in file or file.startswith("DSC")) and Path(file).suffix.lower() == ".jpg":
                    list_to_copy.append(
                        (Path(d_path) / Path(file), TARGET_DIR / "{}_{}.jpg".format(Path(d_path).parts[-1], "RGB"))
                    )
                elif "TD_IR_A_4" in file and Path(file).suffix.lower() == ".jpg":
                    list_to_copy.append(
                        (Path(d_path) / Path(file), TARGET_DIR / "{}_{}.jpg".format(Path(d_path).parts[-1], "IR"))
                    )

    print('Copy {} file(s) to {}'.format(len(list_to_copy), TARGET_DIR))
    for src, dest in tqdm(list_to_copy):
        copyfile(src, dest)
except Exception as ex:
    print("Failed cause of exception - ", ex)
    print(traceback.format_exc())
finally:
    print("Removing the tmp directory - ", TMP_DIR.name)
    rmtree(TMP_DIR, ignore_errors=True)
