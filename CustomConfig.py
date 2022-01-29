# %%
class CustomConfig:
    """클래스의 설정값을 따로 관리하기 위해 만든 클래스.
    클래스의 설정값을 일괄적으로 대입시키기 위한 __call__함수를 정의하고
    클래스의 설정값을 확인하기 쉽게 __repr__함수를 정의하였다.
    단, private한 변수는 확인할 수 없다.
    보통 이 클래스를 상속해서 사용하는 것을 권장한다. 그대로 사용해도 무방하긴 하다.
    """
    
    def __init__(self) -> None:
        self.ignore = []
    
    def __call__(self, **kargs):
        """.set() 메서드의 설명을 참고하길 바란다.
        """
        self.set(**kargs)
    
    def set(self, **kargs):
        """예를 들어, 
        ".set(a=7, b=8)"
        라고 함수를 호출했을 때, 입력한 변수대로 값을 대입해준다.
        """
        for name, value in kargs.items():
            setattr(self, name, value)
            
    def __repr__(self) -> str:
        """해당 클래스에 속한 변수명, 변수값을 출력함.

        Returns:
            str: [(변수명 = 변수값) 형태로 출력]
        """
        
        str_direction = 'This is consisted of :'
        for var, val in self.directions():
            str_direction += f'\n\t{var} = {val if (val is not None) else "Null"}'
        
        return str_direction
    
    def directions(self):
        """
        해당 클래스의 변수들을 (변수명, 변수값)의 형태로 yield 한다.
        단, "_"와 "__"로 시작하거나 "ignore"라는 리스트에 속한 변수는 제외를 시켜 출력한다.

        Yields:
            [(변수명:str, 변수값:object) : tuple]: [(클래스 내의 변수명, 위 변수의 변수값)]
        """
        
        for str_var, value in self.__dict__.items():
            if str_var.startswith(("_", "__")):
                pass
            elif str_var in self.ignore:
                pass
            elif str_var == 'ignore':
                pass
            else:
                yield (str_var, value)


# %%
if __name__ == "__main__":
    cfg = CustomConfig()
    
    cfg(a=1, b=2, c=3)
    print(cfg)
    
    cfg(a=2)
    print(cfg)
    
    cfg.ignore.append("b")
    print(cfg)
    
    cfg(_m=7)
    print(cfg)
    print(cfg._m)
    
    cfg(_m=8)
    print(cfg)
    print(cfg._m)

# %%
if __name__ == '__main__':
    cfg.ignore.clear()
    print(cfg)
    
    del cfg.c
    print(cfg)

# %%



