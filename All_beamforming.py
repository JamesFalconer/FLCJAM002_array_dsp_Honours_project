import Beamform_das_final as delay
import Beamform_phase_final as phase
import Null_and_beam_final
import Null_x
import Null_y
import Null_both
import Get_angle
filename = "1.5Hz_90_65_0.wav"
az_angle = 90
el_angle = 60
tone = 1500
null_ang = 90
f1 = ["1.5Hz_90_65_6.wav","1.5Hz_130_70_10.wav","1.5Hz_115_135_7.wav","1.5Hz_90_50_7.wav","1.5kHz_90_90songHz_135_90_18.wav"]
angs = [[90,65],[130,70],[115,135],[90,50],[135,90]]
for num in range(len(f1)):
    print(Get_angle.run(f1[num], tone))
    Null_both.run(f1[num],angs[num][0],angs[num][1],1500)

"""print(Get_angle.run(filename,tone))
print()
delay.run(filename,az_angle,el_angle,tone)
print()
phase.run(filename,az_angle,el_angle,tone)
print()
Null_x.run(filename,az_angle,tone)
print()
Null_y.run(filename,el_angle,tone)
print()
Null_both.run(filename,az_angle,el_angle,tone)
print()
Null_and_beam_final.run(filename,az_angle,null_ang,tone)
"""
