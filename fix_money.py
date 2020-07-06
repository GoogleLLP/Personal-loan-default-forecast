# 去掉字符串的指定字符
def fix_money(my_str, drop_str="$,"):
    drop_ch = list(drop_str)
    for char in drop_ch:
        my_str = my_str.replace(char, '')
    return eval(my_str)


if __name__ == '__main__':
    print(fix_money("$4,100"))