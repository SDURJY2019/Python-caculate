import numpy as np
import matplotlib.pyplot as plt

def turefft(y,f):
    fft_y = np.fft.rfft(y)
    g=np.abs(fft_y/f)
    g[0]=0
    normalization_y = g*2 # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(f / 2))]  # 由于对称性，只取一半区间（单边频谱）
    return normalization_half_y
def reversefft(y,f):
    fft_y=np.conj(np.fft.fft(y))
    g = np.abs(fft_y / f)
    g[0]=0
    normalization_y = g*2 # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(f / 2))]  # 由于对称性，只取一半区间（单边频谱）
    return normalization_half_y

f=129000
maxnumber1=0
print(1/(2*np.pi/f))
x = np.arange(0, 1, 1/f)
y1 =np.sin(2 * np.pi * 200 * x)+20*np.random.normal(size=f)
y2 =np.sin(2 * np.pi * 200 * x)+20*np.random.normal(size=f)
#y1 =np.sin(2 * np.pi * 200 * x) +np.sin(2 * np.pi * 400 * x) +np.sin(2 * np.pi * 600 * x)+1*np.random.normal(size=f)+np.cos(2*np.pi*800*x)
#y2 =np.sin(2 * np.pi * 200 * x) +np.sin(2 * np.pi * 400 * x) +np.sin(2 * np.pi * 600 * x)+1*np.random.normal(size=f)+np.cos(2*np.pi*800*x)
p1=turefft(y1,f)
p2=reversefft(y2,f)
final=[]
for i in range(len(p1)):
    sum=p1[i]*p2[i]
    final.append(sum)
final[0]=0
half_x = np.linspace(0, f // 2, f // 2 )
plt.plot(x[:50],y1[:50],'r')
plt.xlabel(u"时间(s)", fontproperties='FangSong')
plt.title(u"两信号波形", fontproperties='FangSong')
plt.plot(x[:50],y2[:50],'b')
plt.show()
plt.xlabel(u"频率(Hz)", fontproperties='FangSong')
plt.title(u"信号一傅里叶展开后的波形", fontproperties='FangSong')
plt.plot(half_x[:1000],p1[:1000],'b')
plt.show()
plt.xlabel(u"频率(Hz)", fontproperties='FangSong')
plt.title(u"信号二傅里叶展开后的波形", fontproperties='FangSong')
plt.plot(half_x,p2,'y')
plt.show()
plt.xlabel(u"频率(Hz)", fontproperties='FangSong')
plt.title(u"两信号进行互相关后的频域图", fontproperties='FangSong')
plt.plot(half_x[:1000],final[:1000],'r')
plt.show()