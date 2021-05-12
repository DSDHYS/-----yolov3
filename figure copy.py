import matplotlib.pyplot as plt#约定俗成的写法plt
#首先定义两个函数（正弦&余弦）
import numpy as np

x_1=np.linspace(1,51,51)#-π to+π的256个值
x_2=np.linspace(1,51,51)
x_3=np.linspace(1,51,51)
x_4=np.linspace(1,51,51)
x_5=np.linspace(1,51,51)
x_6=np.linspace(1,51,51)
lr_1=[6716026.2660063645, 
    4009093.4951293943, 
    130300.48340344922, 
    8254.512865375651, 
    4319.324661254883, 
    13369.459104774738, 
    2846.2292991769723, 
    41332.1719049388, 
    52649.93798225666, 
    4841.917597685189, 
    2634.0311253383243, 
    15407.809664969609, 
    6326.464753407446, 
    976.978492690777, 
    451.50475628293793, 
    326.82327840081575, 
    254.25483337270802, 
    377.9345777971991, 
    744.6212398660595, 
    16312.86213744591, 
    73063.062460485, 
    19723.239368162485, 
    1876.5335602990513, 
    1199.8472005515264, 
    1187.1173342211493, 
    1090.429534885801, 
    1356.7973424188021, 
    692.9761039733887, 
    650.2394818010001, 
    712.2229711861446, 
    357.9972117983062, 
    616.8379355200406, 
    577.4995053521518, 
    566.2223272784003, 
    721.8412144159456, 
    628.5459963173702, 
    277.06410502729744, 
    245.35515530684899, 
    313.1128179221318, 
    450.004339967925, 
    360.65523609292916, 
    490.9919353024713, 
    341.99941352318075, 
    186.16013186882282, 
    155.0494958343185, 
    204.85907975558578, 
    265.4429225658548,
    255.895109202944, 
    208.476481765715, 
    235.36732919627224, 
    1583.5467992979904]
lr_2=[1567.9087411946264,
    574.2705823043298,
    82.2376365858933, 
    128.04728882888267, 
    77.1262061875442, 
    59.28323107752307,
    55.65489758458631, 
    56.97662083855991, 
    52.1420852069197,
    56.35010815982161, 
    49.33057418691701, 
    48.389489426284,
    47.60007313695447, 
    43.41655673980713, 
    47.53567643001162, 
    42.627578117107525, 
    43.43808360264219, 
    41.502408968169114, 
    39.38831335922767, 
    38.74065990119145,
    35.685296308583226, 
    31.34030896219714, 
    30.25993891748889,
    28.625444895645668, 
    27.092514745120344, 
    25.838178029553642, 
    25.200555742198024, 
    24.87501720231155, 
    24.318691398357522, 
    23.10953187942505, 
    23.044082690929546, 
    23.087513272515658, 
    23.3259988883446, 
    23.349932825154273, 
    23.474899068372004, 
    23.756158667597276, 
    23.022659734199788, 
    22.854557698348472, 
    21.937888502252513, 
    22.14740144795385, 
    22.1898012786076, 
    21.916085052490235, 
    22.19395464535417, 
    21.989009331012593, 
    21.758238345179066, 
    22.041972396291534, 
    22.456244018160064, 
    21.98636876796854, 
    22.200594226245222, 
    21.885539689557305, 
    21.135963268937736]
lr_3=[226.7124939918518, 
        83.54851840446736, 
        100.4729681771377, 
        31.565172159260715, 
        22.26587207728419, 
        21.45409447242474,
        20.4816322721284,
        20.684875886193637, 
        20.265593791830128, 
        22.495818236778522, 
        25.811139616472968, 
        20.348967690303407, 
        19.511948322427685, 
        18.8643808858148, 
        18.908371224896662, 
        17.619769645559376, 
        18.25234241814449, 
        17.943949222564697, 
        17.74718481590008, 
        18.095685801012763, 
        18.01419345757057, 
        18.38736993197737, 
        18.02197214981605, 
        18.132691965432002, 
        18.02603674592643, 
        17.43229194509572, 
        17.80339580239921, 
        17.444803833139353, 
        17.15137737372826, 
        17.866183747916384, 
        17.232529038396375, 
        18.036716773592193, 
        17.21809588958477, 
        16.968199016307963,
        17.59024660176244, 
        17.600621705219663, 
        17.164870956026274, 
        17.324931971780185, 
        16.82012134420461, 
        16.80273217826054, 
        17.558499615767907,
        17.538300316909265, 
        17.432818095437412, 
        17.10721283616691, 
        17.117463310833635,
        17.19321882971402, 
        16.896370567124464, 
        17.430160933527453, 
        17.079341733866723, 
        16.539089459386364, 
        16.902573205684792]
lr_4=[530.2907372836409, 
    43.5075413144868,
    28.674594533854517, 
    24.427142981825202, 
    21.732253133839574, 
    20.652923426134834, 
    19.33346915409483, 
    19.030030296588766, 
    18.973633394570186,
    17.29298340369915,
    17.4199746904702,
    17.792591407381256, 
    17.194244986567004, 
    16.7172091911579, 
    17.058255738225476, 
    16.350103033000025, 
    15.853280863268623, 
    15.99797531982948, 
    15.937162955053921, 
    15.612704017244535, 
    15.541372868110393, 
    15.896237320735537, 
    15.755445842085214, 
    15.135961437225342, 
    14.862672126704249, 
    15.667614650726318, 
    15.552071857452393,
    14.860330094962285, 
    15.212606370860133,
    14.839648697294038, 
    15.140231157171316,
    14.901826033098944, 
    14.878118815915338, 
    14.834641269157673, 
    14.374969732350317, 
    14.202995306870033, 
    14.682992754311398, 
    14.142075880642595, 
    14.295433716938414, 
    14.121328880046976, 
    14.576319119025921, 
    14.407731368623931, 
    14.061563251758443, 
    13.549613357412404, 
    13.812976707261184, 
    14.007913191565152, 
    13.375084872081361, 
    13.68094592258848, 
    13.529275991176737,
    14.021076153064596, 
    13.853562133065585]
lr_5= [3074.06409744,
 1186.13965822,
 673.85537546,
 419.61152128,
 298.88775879,
 232.44951948,
 185.18112555,
 152.83776479,
 131.81289415,
 114.3878245 ,
 101.99056044,
 93.00827272,
 82.47386968,
 76.46669392,
 71.84798435,
 67.53825728,
 63.18294004,
 58.41567207,
 57.18680815,
 54.4768255 ,
 51.62642814,
 48.52794191,
 47.54626357,
 45.59300494,
 44.50053917,
 44.09529565,
 41.24060137,
 40.71716321,
 40.18336017,
 39.14752154,
 38.14417369,
 37.95229966,
 36.60627182,
 35.93939686,
 34.93460194,
 35.68363669,
 33.8894468 , 
 33.90286892,
 33.26103436,
 33.33015984,
 31.3696833 , 
 32.68118695,
 31.73694163,
 31.01453135,
 31.4745694 ,
 31.59298301,
 30.398753  , 
 30.87204658,
 30.13524635,
 30.3447968 , 
 30.10503742]
lr_6=[8193.87163002,
    7586.08943292,
    7163.76215231,
    6704.21145946,
    6292.52500926,
    5968.89410358,
    5674.59433594,
    5561.53731142,
    5282.10787817,
    5085.3803029 , 
    4999.06942307,
    4841.82463168,
    4635.69729299,
    4707.91224997,
    4587.14318006,
    4587.48528926,
    4472.80101697,
    4441.47905273,
    4435.49333033,
    4380.35937626,
    4288.11156469,
    4344.0401283 , 
    4281.45436422,
    4221.57817299,
    4228.37809638,
    4217.37942273,
    4219.72267393,
    4210.37378014,
    4221.07755927,
    4169.32504799,
    4084.94670494,
    4096.70771905,
    4160.73313746,
    4070.97845712,
    4148.76288473,
    4116.77212251,
    4089.15582486,
    4064.16594491,
    4111.4169362 , 
    4063.27800903,
    4134.660493  ,
    4051.89360646,
    4029.41775828,
    4050.33696331,
    4055.20827721,
    4088.26404272,
    4059.74493366,
    4045.65042767,
    4066.00258621,
    4075.494112  ,
    4068.54159314]
print(x_1)
print(lr_1)
# x_1=np.array(x_1).reshape(1,51)
# x_2=np.array(x_2).reshape(1,51)
# x_3=np.array(x_3).reshape(1,51)
# x_4=np.array(x_4).reshape(1,51)
# x_5=np.array(x_5).reshape(1,51)
# x_6=np.array(x_6).reshape(1,51)
# lr_1=np.array(lr_1).reshape(1,51)
# lr_2=np.array(lr_2).reshape(1,51)
# lr_3=np.array(lr_3).reshape(1,51)
# lr_4=np.array(lr_4).reshape(1,51)
# lr_5=np.array(lr_5).reshape(1,51)
# lr_6=np.array(lr_6).reshape(1,51)
#在ipython的交互环境中需要这句话才能显示出来
fig,axes=plt.subplots(nrows=2,ncols=3)
plt.subplot(231)
fig1=plt.plot(x_1,lr_1,label='c',color='g')
plt.subplot(232)
fig2=plt.plot(x_2,lr_2,label='c',color='black')
plt.subplot(233)
fig3=plt.plot(x_3,lr_3,label='c',color='red')
plt.subplot(234)
fig4=plt.plot(x_4,lr_4,label='c',color='yellow')
plt.subplot(235)
fig5=plt.plot(x_5,lr_5,label='c',color='blue')
plt.subplot(236)
fig6=plt.plot(x_6,lr_6,label='c',color='grey')



plt.show()