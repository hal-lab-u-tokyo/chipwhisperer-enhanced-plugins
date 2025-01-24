import xml.etree.ElementTree as ET

class ParseError(Exception):
    pass

class MemoryRange():
    def __init__(self, base, high):
        self._base = base
        self._high = high

    def getSize(self):
        return self._high - self._base + 1

    @property
    def base(self):
        return self._base

    @property
    def high(self):
        return self._high

    def __str__(self):
        return f"MemoryRange({self._base:X}, {self._high:X}) Size: {self.getSize()//1024}KB"


class MemoryMap():
    def __init__(self):
        self._key_names = []

    def add_range(self, name, base, high):
        setattr(self, name, MemoryRange(base, high))
        self._key_names.append(name)

    def get(self, name):
        return getattr(self, name)

    def getModules(self):
        return iter(self._key_names)

def vivado_parse_memmap(hwh_file, module_name):
    root = ET.parse(hwh_file).getroot()
    modules = root.find("MODULES")
    if modules is None:
        raise ParseError("<MODULES> Tag not found")
    for module in modules.findall("MODULE"):
        if module.get("FULLNAME") == module_name:
            interface = module
            break
    else:
        raise ParseError(f"<MODULE> Tag associated with {module_name} is not found")
    memmap_element = interface.find("MEMORYMAP")
    if memmap_element is None:
        raise ParseError("<MEMORYMAP> Tag not found")
    memmap = MemoryMap()
    for mem in memmap_element.findall("MEMRANGE"):
        name = mem.get("INSTANCE")
        base_value_str = mem.get("BASEVALUE")
        try:
            if base_value_str.startswith("0x"):
                base_value = int(base_value_str, 16)
            else:
                base_value = int(base_value_str)
            high_value_str = mem.get("HIGHVALUE")
            if high_value_str.startswith("0x"):
                high_value = int(high_value_str, 16)
            else:
                high_value = int(high_value_str)
        except ValueError:
            raise ParseError("Invalid address value")

        memmap.add_range(name, base_value, high_value)

    return memmap
