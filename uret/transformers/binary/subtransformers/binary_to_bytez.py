import lief
import array


def binary_to_bytez(binary, dos_stub=False, imports=False):
    # Write modified binary to disk
    builder = lief.PE.Builder(binary)
    builder.build_imports(imports)
    builder.build()

    bytez = array.array("B", builder.get_build()).tobytes()
    return bytez
