from pathlib import Path
import laspy


def extract_ply_fields(filepath):
    xyz_fields = {'x', 'y', 'z'}
    rgb_fields = {'red', 'green', 'blue'}
    path = Path(filepath)
    result = []

    fields = []
    if path.suffix.lower() == ".las":
        las = laspy.read(path)
        fields = [str(name) for name in las.point_format.dimension_names]
    else:
        with open(path, 'rb') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                if line.startswith("property"):
                    parts = line.split()
                    if len(parts) == 3:
                        _, dtype, name = parts
                        fields.append(name)
                elif line.startswith("end_header"):
                    break

    # Group fields
    if xyz_fields.issubset(fields):
        result.append("xyz")
    if rgb_fields.issubset(fields):
        result.append("rgb")
    for name in fields:
        if name not in xyz_fields and name not in rgb_fields:
            result.append(name)

    return result
