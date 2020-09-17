from tools.xml_parser import parser
from tools.converter import Converter
from tools.transformer import transformer
import time
from tools.cfg import *
if __name__ == '__main__':
    start_time=time.time()
    # pase = parser()
    # print('将原图变换为正方形图片，并写入...')
    # conver = Converter(pase)
    # conver()
    print('将原图变为{}*{}...'.format(IMAGE_SIZE,IMAGE_SIZE))
    trans=transformer()
    trans()
    end_time=time.time()
    print("耗时 %.5f"%(end_time-start_time))
