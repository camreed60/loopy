from dynamixel_lib import Dynamixel, U2D2, XM430W210
import numpy as np

# Constants
POSITION_CONTROL = 3
VELOCITY_CONTROL = 1
CURRENT_CONTROL = 0

class Loopy:
    def __init__(self, device_path1='/dev/ttyUSB0', device_path2 = 'dev/ttyUSB1', baudrate=4000000):
        self.u2d2_1 = U2D2(device_path1, baudrate)
        self.u2d2_2 = U2D2(device_path2, baudrate)

        self.motors = [Dynamixel(XM430W210, i, self.u2d2_1) for i in range(0,18)]
        self.motors.extend([Dynamixel(XM430W210, i, self.u2d2_2) for i in range(18,36)])
        
        self.ctrl_mode = CURRENT_CONTROL

        self.DEFAULT_KP = 800
        self.DEFAULT_KD = 0
        self.DEFAULT_KI = 0
        
        self.set_current_control()
        self.set_torque([0.0] * 36)

    @staticmethod
    def bytes_to_int(byte_list):
        return int.from_bytes(byte_list, byteorder='little', signed=False)

    @staticmethod
    def bytes_to_twos_complement(byte_list):
        return int.from_bytes(byte_list, byteorder='little', signed=True)

    def read_position(self):

        read_data_0 = self.u2d2_1.bulk_read(range(18), XM430W210.PresentPosition)
        read_data_1 = self.u2d2_2.bulk_read(range(18,36), XM430W210.PresentPosition)
        
        positions = [0]*36
        for i in range(36):
            if i < 18:
                data = read_data_0[i][XM430W210.PresentPosition]
            else:
                data = read_data_1[i][XM430W210.PresentPosition]

            if self.ctrl_mode == POSITION_CONTROL:
                positions[i] = (self.bytes_to_int(data) % 4096) * 2 * np.pi / 4096.0 - np.pi
            else:
                positions[i] = self.bytes_to_twos_complement(data)*0.087891*np.pi/180.0 - np.pi
    
        return positions
        
    def read_velocity(self):
        
        read_data_0 = self.u2d2_1.bulk_read(range(18), XM430W210.PresentVelocity)
        read_data_1 = self.u2d2_2.bulk_read(range(18,36), XM430W210.PresentVelocity)
        
        velocities = [0]*36
        for i in range(36):
            if i < 18:
                data = read_data_0[i][XM430W210.PresentVelocity]
            else:
                data = read_data_1[i][XM430W210.PresentVelocity]

            velocities[i] = self.bytes_to_twos_complement(data) * 0.229 * 2 * np.pi / 60.0  # rad/s
        
        return velocities

    def read_torque(self):
        
        read_data_0 = self.u2d2_1.bulk_read(range(18), XM430W210.PresentCurrent)
        read_data_1 = self.u2d2_2.bulk_read(range(18,36), XM430W210.PresentCurrent)
        
        torques = [0]*36
        for i in range(36):
            if i < 18:
                data = read_data_0[i][XM430W210.PresentCurrent]
            else:
                data = read_data_1[i][XM430W210.PresentCurrent]

            torques[i] = self.bytes_to_twos_complement(data) * 0.00269  # Nm
        
        return torques


    def set_torque(self, torques):
        
        if len(torques) != len(self.motors):
            print("Wrong number of torques provided")
            return

        max_torque = 1.5  # Nm
        torques = np.clip(torques, -max_torque, max_torque)
     

        data = [int(t / 0.00269) for t in torques]
        fields = [XM430W210.GoalCurrent]
        
        self.u2d2_1.bulk_write(range(18), fields, data[:18])
        self.u2d2_2.bulk_write(range(18,36),fields,data[18:])
        


       

        # for i, (motor, torque) in enumerate(zip(self.motors, torques)):
        #     if torque > max_torque:
        #         torque = max_torque
        #         print(f"Torque too high for motor {i}, setting to max")
        #     elif torque < -max_torque:
        #         torque = -max_torque
        #         print(f"Torque too low for motor {i}, setting to min")

        #     current = int(torque / 0.00269)
        #     motor.write(XM430W210.GoalCurrent, current)

    def set_position(self, positions):

        if len(positions) != len(self.motors):
            print("Wrong number of positions provided")
            return
        
        data = [int((p + np.pi) * 4096 / (2 * np.pi)) % 4096 for p in positions]
        fields = [XM430W210.GoalPosition]
        
        self.u2d2_1.bulk_write(range(18), fields, data[:18])
        self.u2d2_2.bulk_write(range(18,36),fields,data[18:])
        

    def set_Kp(self, Kps):
        
        if len(Kps) != len(self.motors):
            print("Wrong number of Kps provided")
            return

        data = [int(Kp) for Kp in Kps]
        fields = [XM430W210.PositionPGain]

        self.u2d2_1.bulk_write(range(18), fields, data[:18])
        self.u2d2_2.bulk_write(range(18,36),fields,data[18:])
    
    def set_Kd(self, Kds):
        if len(Kds) != len(self.motors):
            print("Wrong number of Kds provided")
            return

        data = [int(Kd) for Kd in Kds]
        fields = [XM430W210.PositionDGain]

        self.u2d2_1.bulk_write(range(18), fields, data[:18])
        self.u2d2_2.bulk_write(range(18,36),fields,data[18:])
    
    def set_Ki(self, Kis):
        if len(Kis) != len(self.motors):
            print("Wrong number of Kis provided")
            return

        data = [int(Ki) for Ki in Kis]
        fields = [XM430W210.PositionIGain]

        self.u2d2_1.bulk_write(range(18), fields, data[:18])
        self.u2d2_2.bulk_write(range(18,36),fields,data[18:])

    def reset_gains(self):
        self.set_Kp([self.DEFAULT_KP] * 36)
        self.set_Kd([self.DEFAULT_KD] * 36)
        self.set_Ki([self.DEFAULT_KI] * 36)

    def set_position_control(self):

        for motor in self.motors:
            motor.write(XM430W210.TorqueEnable, 0)
            motor.write(XM430W210.OperatingMode, POSITION_CONTROL)
            motor.write(XM430W210.TorqueEnable, 1)
        
        self.ctrl_mode = POSITION_CONTROL

    def set_current_control(self):
        
        for motor in self.motors:
            motor.write(XM430W210.TorqueEnable, 0)
            motor.write(XM430W210.OperatingMode, CURRENT_CONTROL)
            motor.write(XM430W210.TorqueEnable, 1)
        
        self.ctrl_mode = CURRENT_CONTROL
