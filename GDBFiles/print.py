import gdb


class pstdVector:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        res = "\npstd::vector {\n"
        res += "    nStored = {}\n\n".format(self.val["nStored"])
        value = self.val["ptr"]
        for index in range(self.val["nStored"]):
            res += "    [{}] = {}\n".format(index, value.dereference())
            value += 1

        res += "}"
        return res

    def pstdvector_lookup_function(val):
        if val.type.code == gdb.TYPE_CODE_PTR:
            return None # to add
        if "pstd::vector<float, pstd::pmr::polymorphic_allocator<float> >" == val.type.tag:
            return pstdVector(val)
        return None

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    pstdVector.pstdvector_lookup_function,
    replace = True
)
