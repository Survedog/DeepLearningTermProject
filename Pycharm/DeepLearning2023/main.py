# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from konlpy.tag import Kkma

parser = Kkma()

if __name__ == '__main__':
    words = parser.morphs('안녕하세요. 저는 프로그래머입니다. 만나서 반갑습니다.')
    print(words)
