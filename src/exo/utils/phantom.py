class _PhantomData[*T]:
    """
    此說明已翻譯為繁體中文。
    """


type PhantomData[*T] = _PhantomData[*T] | None
"""
Allows you to use generics in functions without storing anything of that generic type. 
Just use `None` and you'll be fine
"""
