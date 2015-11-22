#!/usr/bin/env python
"""
Just plot
"""
import matplotlib.pyplot as plt
values = range(2,101)

N=20
results = [0.9966, 0.937, 0.839975, 0.622175, 0.34235, 0.1932, 0.11095, 0.080825, 0.060225, 0.05125, 0.04145, 0.037025, 0.032275, 0.03025, 0.027025, 0.025225, 0.024, 0.021525, 0.021625, 0.01995, 0.018475, 0.01865, 0.0175, 0.017125, 0.01605, 0.014775, 0.015625, 0.015325, 0.014525, 0.013975, 0.013225, 0.013575, 0.012975, 0.012675, 0.0121, 0.012225, 0.012375, 0.01235, 0.0114, 0.011525, 0.011825, 0.011625, 0.01125, 0.0108, 0.010625, 0.0106, 0.010525, 0.010275, 0.01015, 0.0102, 0.009525, 0.010275, 0.00955, 0.009525, 0.00965, 0.009375, 0.0094, 0.009425, 0.009375, 0.00935, 0.00905, 0.008975, 0.0088, 0.008975, 0.0088, 0.0088, 0.00875, 0.008675, 0.008375, 0.008575, 0.0088, 0.008175, 0.008425, 0.008175, 0.00825, 0.0082, 0.007975, 0.0083, 0.008075, 0.008225, 0.008025, 0.007925, 0.00775, 0.007925, 0.00815, 0.007975, 0.00785, 0.007525, 0.007825, 0.007475, 0.007975, 0.00755, 0.007475, 0.00775, 0.0075, 0.0075, 0.007475, 0.007525, 0.007625]

stds = [0.0189457119159, 0.12577161842, 0.175884015405, 0.230375143787, 0.162342931783, 0.0934826721912, 0.0505603352441, 0.0351568894386, 0.0215554372491, 0.019389107767, 0.0155931876151, 0.00885081210963, 0.00914531984132, 0.00792543374207, 0.00633141966703, 0.00645266417846, 0.00657647321898, 0.00463539372653, 0.00494184935019, 0.00424234604906, 0.00355132017706, 0.00402212630334, 0.00335410196625, 0.00400585508974, 0.00337601836488, 0.00318384594477, 0.00374791608764, 0.0032337091706, 0.00307601609229, 0.00264799452416, 0.00262904450324, 0.00281191660616, 0.00315624381188, 0.00274442616953, 0.00231084400166, 0.00242113093409, 0.00312999600639, 0.00295423424934, 0.0024062418831, 0.00242113093409, 0.00268921456935, 0.00250935748749, 0.0024366985862, 0.00179861057486, 0.00263094564748, 0.00237486841741, 0.00232634369774, 0.0023424079491, 0.00199436706752, 0.00214126131054, 0.00164677715554, 0.00208851023459, 0.00191637678967, 0.00186061145863, 0.00223662692463, 0.00207289049397, 0.0017, 0.0020265426223, 0.00232849200127, 0.00208026440627, 0.00168745370307, 0.00177112252541, 0.00185337529929, 0.00193955536142, 0.00188679622641, 0.00167630546142, 0.00188745860882, 0.00163764312352, 0.0017809758561, 0.00197658164516, 0.00204572725455, 0.00169022927439, 0.00160682139642, 0.0014515078367, 0.00163935963108, 0.0018398369493, 0.00152868407462, 0.00176351920885, 0.00165283846761, 0.00177816619021, 0.00181297407593, 0.00169760861214, 0.0015612494996, 0.00183898749316, 0.00156604597634, 0.00148723737177, 0.00158192920196, 0.00147880864212, 0.00148555545167, 0.00129879752079, 0.00186061145863, 0.00158034806293, 0.00129879752079, 0.00175, 0.00136930639376, 0.00158113883008, 0.00163916899678, 0.00156104932657, 0.00163458710383]


#N=30
#results=[0.998777777778, 0.998722222222, 0.9714, 0.844644444444, 0.590566666667, 0.2411, 0.104811111111, 0.0577, 0.0449, 0.0326777777778, 0.0283, 0.0250888888889, 0.0215111111111, 0.0197444444444, 0.0180111111111, 0.0156111111111, 0.0148222222222, 0.0141444444444, 0.0130333333333, 0.0123333333333, 0.0119111111111, 0.0113222222222, 0.0106111111111, 0.0104333333333, 0.0097, 0.00958888888889, 0.00911111111111, 0.00883333333333, 0.00842222222222, 0.00831111111111, 0.00808888888889, 0.00785555555556, 0.00776666666667, 0.00735555555556, 0.00731111111111, 0.00712222222222, 0.00698888888889, 0.00701111111111, 0.00675555555556, 0.00668888888889, 0.00665555555556, 0.00637777777778, 0.00623333333333, 0.0061, 0.00591111111111, 0.00597777777778, 0.00593333333333, 0.00574444444444, 0.0056, 0.00564444444444, 0.00544444444444, 0.00538888888889, 0.00541111111111, 0.00528888888889, 0.00543333333333, 0.00533333333333, 0.00537777777778, 0.00512222222222, 0.00503333333333, 0.00508888888889, 0.00491111111111, 0.00487777777778, 0.00484444444444, 0.00495555555556, 0.00488888888889, 0.00477777777778, 0.00461111111111, 0.00476666666667, 0.00468888888889, 0.00458888888889, 0.00478888888889, 0.00455555555556, 0.00463333333333, 0.00442222222222, 0.00447777777778, 0.00436666666667, 0.00437777777778, 0.00442222222222, 0.00445555555556, 0.00432222222222, 0.00433333333333, 0.00427777777778, 0.0042, 0.00424444444444, 0.00421111111111, 0.00418888888889, 0.00421111111111, 0.0042, 0.00412222222222, 0.00404444444444, 0.00405555555556, 0.00385555555556, 0.00394444444444, 0.00401111111111, 0.00404444444444, 0.00393333333333, 0.00393333333333, 0.00383333333333, 0.00391111111111]

#stds=[0.0053874568544, 0.00365950479008, 0.0558871481947, 0.157367885409, 0.246108351878, 0.170900916713, 0.0731666548553, 0.0190491599244, 0.0153334098227, 0.00734190576396, 0.00668599974518, 0.00623645198276, 0.00429331953069, 0.00447725389822, 0.00392300904919, 0.00255977911856, 0.0030640085098, 0.00262398716498, 0.00208377032454, 0.00244696839395, 0.00230447765968, 0.00201656718457, 0.00168049816247, 0.00167696816375, 0.00190421883793, 0.00176841283376, 0.00143156652519, 0.00168782861455, 0.00147556224898, 0.00135610189355, 0.00133407386843, 0.00143376393188, 0.00146981984283, 0.00123508384281, 0.00139823521927, 0.00144952099368, 0.00133606202266, 0.0012736164122, 0.00124404755576, 0.000968389269756, 0.00124221029611, 0.000924561958343, 0.00119561957697, 0.000974932729557, 0.00106434933465, 0.00110866397185, 0.00113550989254, 0.000956524056992, 0.00102944309528, 0.000856060457078, 0.00098757715748, 0.000986013297183, 0.000976956725984, 0.000833259255967, 0.000844663714222, 0.00095581391856, 0.000925896295822, 0.000901233722306, 0.000808214004174, 0.000891385383117, 0.000772681717809, 0.00082991893104, 0.000839164781711, 0.000921887489453, 0.000888888888889, 0.000808901098809, 0.000709285851934, 0.000906696622599, 0.000924027683835, 0.000764247390623, 0.000856276752308, 0.000761739400045, 0.000832295650225, 0.000753264500297, 0.000761009725607, 0.000613731754651, 0.00085807709228, 0.000845905169363, 0.00086773438503, 0.00082991893104, 0.000777777777778, 0.000880165528764, 0.000779046584131, 0.000791622805803, 0.00087904268917, 0.000700176344631, 0.00087904268917, 0.000840047029842, 0.000741952712213, 0.000778095173334, 0.000691661088777, 0.000637413755967, 0.000616140917023, 0.000718279838173, 0.000657717714975, 0.000727247474309, 0.000727247474309, 0.000635862396792, 0.000693532734859]



plt.ylabel('Percentage of largest cultural domain')
plt.xlabel('Q -Number of traits')
plt.errorbar(x=values, y=results, yerr = stds, ecolor='r', fmt = 'o')

plt.title("Axelrod model- %d X %d Lattice" % (N,N))
plt.show()
