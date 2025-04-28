import numpy as np 
from typing import Optional

class GameState:
    def __init__(self, nodes, p_max, area_size=(5, 5)):
        self.nodes = nodes
        self.p_max = p_max
        self.gamma = 0.01
        self.beta = 1
        self.noise_power = 0.01
        self.area_size = area_size
        self.positions = self.generate_positions()
        self.observation_space = 2*nodes * nodes + nodes  # interferensi, channel gain, power
        self.action_space = nodes
        self.p = np.random.uniform(0, self.p_max, size=self.nodes)
        self.lamda=150 #buat punishment reward data rate dan power
        self.gf = 1.2 #buat growth factor
        self.efsilon=10e-6 # tolerance convergence
        self.x=1
    def sample_valid_power(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)
        return rand * self.p_max

    def reset(self,gain,*, seed: Optional[int] = None, options: Optional[dict] = None):
        power = self.sample_valid_power()
        #super().ini(seed=seed)
        #loc = self.generate_positions()
        #gain= self.generate_channel_gain(loc)
        intr=self.interferensi(power,gain)
        #new_intr=self.interferensi_state(intr)
        #ini_sinr=self.hitung_sinr(ini_gain,intr,power)
        #ini_data_rate=self.hitung_data_rate(ini_sinr)
        #ini_EE=self.hitung_efisiensi_energi(self.p,ini_data_rate)
        gain_norm=self.norm(gain)
        intr_norm = self.norm(intr)
        p_norm=self.norm(power)
        
        result_array = np.concatenate((np.array(gain_norm).flatten(), np.array(intr_norm).flatten(),np.array(p_norm)))
        return result_array ,{}

    def step_function(self,x):
        if x<=0 :
            x= 0
        else :
            x=1
        return x
    def step(self,power,channel_gain,next_channel_gain,episode):
        #for i in range(len(power)):
        #    if power[i]<10e-6:
        #        power[i]=10e-6
        intr=self.interferensi(power,channel_gain)
        convergence=True
        hasil=True
        next_intr=self.interferensi(power,next_channel_gain)
        sinr=self.hitung_sinr(channel_gain,intr,power)
        data_rate=self.hitung_data_rate(sinr)
        data_rate_constraint=[]
        #intr_state=self.interferensi_state(new_intr)
        
        total_daya=np.sum(power)
        power_constraint = total_daya-self.p_max
        for i in range(self.nodes):
            dr=data_rate[i]
            if max(power_constraint,0)>self.efsilon or max(0.51-dr,0)>self.efsilon :
                convergence = False 
                hasil= convergence and hasil
            else : 
                convergence = True
                hasil = convergence and hasil
        
        if hasil==False and episode > self.x :
            self.lamda= self.lamda * self.gf
            self.x=episode
        for i in range(self.nodes):
            data_rate_constraint.append(self.lamda*self.step_function(0.51-data_rate[i]))
        EE=self.hitung_efisiensi_energi(power,data_rate)
        gain_norm=self.norm(next_channel_gain)
        intr_norm = self.norm(next_intr)
        p_norm=self.norm(power)
        result_array = np.concatenate((np.array(gain_norm).flatten(), np.array(intr_norm).flatten(),np.array(p_norm)))
        reward = float(EE -  self.lamda*self.step_function(total_daya-self.p_max)-np.sum(data_rate_constraint))
        
        #if reward > 0:
        #    reward = np.log2(reward)
        #else:
        #    reward = -10
        #print(self.lamda, episode)
        return result_array,reward, False,False,{},EE,data_rate

    def norm(self,x):
        x = np.maximum(x, 1e-10)
        x_log = np.log10(x + 1e-10)  # +1e-10 untuk menghindari log(0)
        x_min = np.min(x_log)
        x_max = np.max(x_log)
        return (x_log - x_min) / (x_max - x_min + 1e-10) 

    def generate_positions(self):
        """Generate random positions for all nodes in 2D space (meter)"""
        loc = np.random.uniform(0, self.area_size[0], size=(self.nodes, self.nodes))
        for i in range (self.nodes) :
            for j in range (self.nodes):
              current = loc[i][j]
              loc[j][i]=current
        return loc
    def generate_channel_gain(self, positions):
        channel_gain = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j]) + 1e-6  # avoid zero
                    path_loss_dB = 128.1 + 37.6 * np.log10(distance / 1000)  # example log-distance PL
                    path_loss_linear = 10 ** (-path_loss_dB / 10)
                    rayleigh = np.random.rayleigh(scale=1)
                    channel_gain[i][j] = path_loss_linear * rayleigh
                else:
                    channel_gain[i][j] = np.random.rayleigh(scale=1)
        return channel_gain
    def interferensi(self, power,channel_gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    interferensi[i][j] = channel_gain[i][j] * power [i]
                else:
                    interferensi[i][j] = 0
        return interferensi
    
    def interferensi_state(self, interferensi):
        interferensi_state = np.zeros(self.nodes)
        for i in range(self.nodes):
            for j in range(self.nodes):
                interferensi_state[i]+=interferensi[j][i]
        return interferensi_state
        
    def hitung_sinr(self, channel_gain, interferensi, power):
        sinr = np.zeros(self.nodes)
        for node_idx in range(self.nodes):
            sinr_numerator = (abs(channel_gain[node_idx][node_idx])) * power[node_idx]
            sinr_denominator = self.noise_power**2 + np.sum([(abs(interferensi[node_idx][i])) for i in range(self.nodes) if i != node_idx])
            sinr[node_idx] = sinr_numerator / sinr_denominator
        return sinr 

    def hitung_data_rate(self, sinr):
        sinr = np.maximum(sinr, 0)  # jika ada yang negatif, dibatasi 0
        return np.log(1 + sinr)

    def hitung_efisiensi_energi(self, power, data_rate):
        """Menghitung efisiensi energi total"""
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        energi_efisiensi=total_rate / total_power if total_power > 0 else 0
        return energi_efisiensi
