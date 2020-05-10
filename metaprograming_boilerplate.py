from inspect import Parameter, Signature
from collections import OrderedDict


def _make_setter(dcls):
    code = 'def __set__(self, instance, value):\n'
    for d in dcls.__mro__:
        if 'set_code' in d.__dict__:
            for line in d.set_code():
                code += ' ' + line + '\n'
    return code


class DescriptorMeta(type):
    def __init__(self, clsname, bases, clsdict):
        if '__set__' not in clsdict:
            code = _make_setter(self)
            exec(code, globals(), clsdict)
            setattr(self, '__set__',
                    clsdict['__set__'])
        else:
            raise TypeError('Define set_code()')


class Descriptor(metaclass=DescriptorMeta):
    def __init__(self, name=None):
        self.name = name

    @staticmethod
    def set_code():
        return [
            'instance.__dict__[self.name] = value'
        ]

    # def __set__(self, instance, value):
    #     instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError("Can't delete")


class Typed(Descriptor):
    ty = object

    @staticmethod
    def set_code():
        return [
            'if not isinstance(value, self.ty):',
            ' raise TypeError("Expected %s"%self.ty)'
        ]


class Integer(Typed):
    ty = int


class Positive(Descriptor):
    @staticmethod
    def set_code():
        return [
            'if value < 0:',
            ' raise ValueError("Expected >= 0")'
        ]


class PosInteger(Integer, Positive):
    pass


def make_signature(names):
    return Signature(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in names)


def add_signature(*names):
    def decorate(cls):
        cls.__signature__ = make_signature(names)
        return cls

    return decorate


def _make_init(fields):
    code = 'def __init__(self, %s):\n' % ', '.join(fields)
    for name in fields:
        code += ' self.%s = %s\n' % (name, name)
    return code


class StructMeta(type):
    @classmethod
    def __prepare__(cls, name, bases):
        return OrderedDict()

    def __new__(mcs, name, bases, clsdict):
        fields = [key for key, val in clsdict.items()
                  if isinstance(val, Descriptor)]
        for name in fields:
            clsdict[name].name = name
        if fields:
            exec(_make_init(fields), globals(), clsdict)  # new

        clsobj = super().__new__(mcs, name, bases, dict(clsdict))
        setattr(clsobj, '_fields', fields)  # new
        # sig = make_signature(fields)
        # setattr(clsobj, '__signature__', sig)
        return clsobj

    def __init__(self, clsname, bases, clsdict):
        tt = 1


class Structure(metaclass=StructMeta):
    _fields = []


class Stock(Structure):
    shares = PosInteger()


if __name__ == '__main__':
    tt = Stock(shares=1)
