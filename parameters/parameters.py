import math

B = 20  # Mhz bandwidth
power_u = 0.1  # uplink power:100 mW

noise_p = 10 ** ((-203.975 + 10 * math.log10(20 * 10 ** 6) + 9) / 10)  # noise power
Pu = power_u / noise_p  # normalized received SNR
Pp = Pu  # pilot power

Hb = 15  # base station height in m
Hm = 1.65  # mobile height in m
f = 1900  # frequency in MHz
aL = (1.1 * math.log10(f) - 0.7) * Hm - (1.56 * math.log10(f) - 0.8)
L = 46.3 + 33.9 * math.log10(f) - 13.82 * math.log10(Hb) - aL
d0 = 0.01  # km
d1 = 0.05  # km
