class Multifloat:
    """
    Multiple floating point precision number
    """
    def __init__(self, std, tiny, factor):
        """
        `std`: standard component of this number
        `tiny`: differential element of this number
        `factor`: the approximate factor of the differential element
        """
        self.std = std
        self.tiny = tiny
        self.factor = factor

    def get_std(self):
        if isinstance(self.std, Multifloat):
            return self.std.flatten()
        else:
            return self.std

    def get_tiny(self):
        if isinstance(self.tiny, Multifloat):
            return self.tiny.flatten()
        else:
            return self.tiny

    def flatten(self):
        flat = self.get_std() + self.get_tiny() * self.factor
        if isinstance(flat, Multifloat):
            return flat.flatten()
        else:
            return flat

    def __add__(self, other):
        """ Add another multifloat or a tensor to this multifloat """
        if isinstance(other, Multifloat):
            if self.factor > other.factor:
                fact = other.factor / self.factor
                return Multifloat(
                    std=self.std + other.std,
                    tiny=self.tiny + other.tiny * fact,
                    factor=self.factor,
                )
            else:
                fact = self.factor / other.factor
                return Multifloat(
                    std=self.std + other.std,
                    tiny=self.tiny * fact + other.tiny,
                    factor=other.factor,
                )
        else:
            return Multifloat(std=self.std + other, tiny=self.tiny, factor=self.factor)

    def __neg__(self):
        return Multifloat(std=-self.std, tiny=-self.tiny, factor=self.factor)

    def __sub__(self, other):
        """ Subtract another multifloat or a tensor from this multifloat """
        if not isinstance(self, Multifloat):
            return -other - self
        if isinstance(other, Multifloat):
            if self.factor > other.factor:
                fact = other.factor / self.factor
                return Multifloat(
                    std=self.std - other.std,
                    tiny=self.tiny - other.tiny * fact,
                    factor=self.factor,
                )
            else:
                fact = self.factor / other.factor
                return Multifloat(
                    std=self.std - other.std,
                    tiny=self.tiny * fact - other.tiny,
                    factor=other.factor,
                )
        else:
            return Multifloat(std=self.std - other, tiny=self.tiny, factor=self.factor)

    def __mul__(self, other):
        """ Multiply this multifloat by a tensor or multifloat """
        if isinstance(other, Multifloat):
            return Multifloat(
                std=self.std * other.std,
                tiny=self.tiny * other.tiny,
                factor=self.factor * other.factor,
            )
        else:
            return Multifloat(
                std=self.std * other, tiny=self.tiny * other, factor=self.factor
            )
    
    __rmul__ = __mul__

    def __rsub__(self, other):
        """
        other - self
        """
        return other + (-self)

    def __truediv__(self, other):
        """ Divide this multifloat by a tensor """
        if isinstance(other, Multifloat):
            raise NotImplementedError()
        else:
            return Multifloat(
                std=self.std / other, tiny=self.tiny / other, factor=self.factor
            )

