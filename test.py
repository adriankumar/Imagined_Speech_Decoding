# from .env_mod.helper_val_functions import recommend_harmonic_order

# # 6 channels bind the joint L, so a set including it cannot share a high L
# r = recommend_harmonic_order([6, 64, 128])
# print(r["recommended_L"], r["unresolved"])   #expect 1 and []

# #the methodology's fixed L=4 needs 25 modes, which the 6-channel cap cannot carry
# r4 = recommend_harmonic_order([6, 64, 128], target_L=4)
# print(r4["modes_required"], r4["unresolved"]) #expect 25 and [6]
# print(r4["per_count"])                         #6 resolves False, 64 and 128 resolve True


# #========================================
# from .env_mod.env_module import EEGEnv

# filepath = ""

# channels_to_exclude = ['10', '11', 'TPP5h', 'P11', 'PO11', 'POO11h', '84', '85', 'POO12h', 'PO12', '110', '111', 'P12']
# exclude_2 = ['FTT9h', 'TTP7h', 'FTT7h', 'FFT7h', 'FCC5h', 'CCP5h', 'PPO7', 
#              'CPP5h', 'FFC5h', 'AFF5h', 'FFC3h', 'FCC3h', 'CCP3h', 'CPP3h', 
#              'POO7', 'POO9h', 'I1', 'OI1', 'POO3', 'PPO1', 'CPP1h', 'CCP1h', 
#              'FCC1h', 'FFC1h', 'FCCz', 'CPPz', 'PPOz', 'POOz', 'I2', 'OI2', 
#              'POO4', 'PPO2', 'CPP2h', 'CCP2h', 'FCC2h', 'FFC2h', 'AFF6h', 'FFC4h', 
#              'FCC4h', 'CCP4h', 'CPP4h', 'POO8', 'POO10h', 'PPO8', 'CPP6h', 'FFC6h', 
#              'FFT8h', 'FCC6h', 'CCP6h', 'TPP8h', 'TTP8h', 'FTT8h', 'FTT10h']

# env = EEGEnv(source=filepath, montage="standard_1005", ref_scheme="average", exclude_chns=channels_to_exclude)
# print(env.get_L)                  #expect 4

# env.change_montage(montage="standard_1020", exclude_chns=exclude_2 + channels_to_exclude)
# print(env.get_L)                  #expect 4 still, no auto-set back to the channel ceiling

from EEGEnv import EEGEnv, UnresolvedChannelsError

filepath = r""
#1) default policy raises a typed error carrying the names, nothing committed
try:
    env = EEGEnv(source=filepath, montage="standard_1005", ref_scheme="average")
except UnresolvedChannelsError as e:
    print("unresolved count:", len(e.unresolved), "->", e.unresolved[:6])

#2) auto_exclude=True self-resolves and records the three disjoint buckets
env = EEGEnv(source=filepath, montage="standard_1005", ref_scheme="average", auto_exclude=True)
print("default :", env.get_default_excluded)
print("auto    :", env.get_auto_excluded)
print("manual  :", env.get_e_chns)
print("resolved:", env.get_n_chns, "| L:", env.get_L)

#3) prompt policy on a montage change, answer y or n at the terminal
env.change_montage("standard_1020", auto_exclude="prompt")
print("after montage change:", env.get_auto_excluded[:6], "| resolved:", env.get_n_chns)