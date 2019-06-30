import matplotlib.pyplot as plt
import random
##样本数据
x_train = [1.1,1.3,1.5,2,2.2,2.9,3,3.2,3.2,3.7,3.9,4,4,4.1,4.5,4.9,5.1,5.3,5.9,6,6.8,7.1,7.9,8.2,8.7,9,9.5,9.6,10.3,10.5]
y_train = [39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940,91738,98273,101302,113812,109431,105582,116969,112635,122391,121872]
#样本个数
m = len(x_train)
#步长
alpha = 0.001
#循环次数
cnt0 = 0
cnt1 = 0
#假设函数为 y=theta0+theta1*x
def h(x):
    return theta00 + theta01*x
def s(x):
    return theta10 + theta11*x
theta00 = 0
theta01 = 0
theta10 = 0
theta11 = 0
#导数
diff0=0
diff1=0
#误差
error00=0           #批量
error01=0          
error10=0           #随机
error11=0
#每次迭代theta的值
retn00 = []         #批量
retn01 = []         
retn10 = []         #随机
retn11 = []
#退出迭代的条件
epsilon=0.00001

#批量梯度下降
while 1:
    cnt0=cnt0+1
    diff0=0
    diff1=0
    #梯度下降
    for i in range(m):
        diff0+=h(x_train[i])-y_train[i]
        diff1+=(h(x_train[i])-y_train[i])*x_train[i]
    theta00=theta00-alpha/m*diff0
    theta01=theta01-alpha/m*diff1
    retn00.append(theta00)
    retn01.append(theta01)
    error01=0
    #计算迭代误差
    for i in range(len(x_train)):
        error01 += ((theta00 + theta01 * x_train[i])-y_train[i]) ** 2 / 2
    #判断是否已收敛
    if abs(error01 - error00) < epsilon:
        break
    else:
        error00 = error01

#随机梯度下降
for i in range(10000):
    cnt1=cnt1+1
    diff0=0
    diff1=0
    j=random.randint(0,m-1)
    diff0+=s(x_train[j])-y_train[j]
    diff1+=(s(x_train[j])-y_train[j])*x_train[j]
    theta10=theta10-alpha/m*diff0
    theta11=theta11-alpha/m*diff1
    retn10.append(theta10)
    retn11.append(theta11)
    error11=0
    #计算迭代的误差
    for i in range(len(x_train)):
        error11 += ((theta10 + theta11 * x_train[i])-y_train[i]) ** 2 / 2
    #判断是否已收敛
    if abs(error11 - error10) < epsilon:
        break
    else:
        error10 = error11
plt.title('BGD')
plt.plot(range(len(retn00)),retn00,label='theta0')
plt.plot(range(len(retn01)),retn01,label='theta1')
plt.legend()          #显示上面的label
plt.xlabel('time')
plt.ylabel('theta')
plt.show()
plt.title('SGD')
plt.plot(range(len(retn10)),retn10,label='theta0')
plt.plot(range(len(retn11)),retn11,label='theta1')
plt.legend()
plt.xlabel('time')
plt.ylabel('theta')
plt.show()
plt.plot(x_train,y_train,'bo')
plt.plot(x_train,[h(x) for x in x_train],color='k',label='BGD')
plt.plot(x_train,[s(x) for x in x_train],color='r',label='SGD')
plt.legend()
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
print("批量梯度下降法：theta0={},theta1={}".format(theta00,theta01))
print("批量梯度下降法循环次数：{}".format(cnt0))
print("随机梯度下降法：theta0={},theta1={}".format(theta10,theta11))
print("随机梯度下降法循环次数：{}".format(cnt1))
plt.show()

