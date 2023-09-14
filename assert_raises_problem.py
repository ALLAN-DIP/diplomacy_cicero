from unittest import TestCase


class InputValidationException(ValueError):
    pass


def always_fail():
    raise InputValidationException


instance = TestCase()
instance.assertRaises(ZeroDivisionError, lambda x: x / 0, 1)
instance.assertRaises(InputValidationException, always_fail)
instance.assertRaises(ValueError, always_fail)
instance.assertRaises(ZeroDivisionError, always_fail)
