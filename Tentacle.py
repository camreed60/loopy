from dynamixel_lib import Dynamixel, U2D2, XM430W210
import numpy as np

# Constants
POSITION_CONTROL = 3
VELOCITY_CONTROL = 1
CURRENT_CONTROL = 0

class Tentacle:
    def __init__(self, device_path='/dev/ttyUSB0', baudrate=4000000, motor_ids=[1,2,3,4]):
        self.u2d2 = U2D2(device_path, baudrate)

        self.motors = [Dynamixel(XM430W210, i, self.u2d2) for i in motor_ids]
        self.ctrl_mode = CURRENT_CONTROL
        self.motor_ids = motor_ids
        
        self.DEFAULT_KP = 800
        self.DEFAULT_KD = 0
        self.DEFAULT_KI = 0

        self.set_current_control()
        self.set_torque([0.0] * len(motor_ids))

    @staticmethod
    def bytes_to_int(byte_list):
        return int.from_bytes(byte_list, byteorder='little', signed=False)

    @staticmethod
    def bytes_to_twos_complement(byte_list):
        return int.from_bytes(byte_list, byteorder='little', signed=True)

    def read_position(self):
        positions = []

        for motor in self.motors:
            read_pos_bytes, _, _ = motor.read(XM430W210.PresentPosition)
            if self.ctrl_mode == POSITION_CONTROL:
                positions.append((self.bytes_to_int(read_pos_bytes) % 4096) * 2 * np.pi / 4096.0 - np.pi)
            else:
                positions.append(self.bytes_to_twos_complement(read_pos_bytes)*0.087891*np.pi/180.0 - np.pi)

        return positions
        

    def read_velocity(self):
        velocities = []
        for motor in self.motors:
            read_vel_bytes, _, _ = motor.read(XM430W210.PresentVelocity)
            velocities.append(self.bytes_to_twos_complement(read_vel_bytes) * 0.229 * 2 * np.pi / 60.0)  # rad/s
        
        return velocities

    def read_torque(self):
        torques = []
        for motor in self.motors:
            read_cur_bytes, _, _ = motor.read(XM430W210.PresentCurrent)
            torques.append(self.bytes_to_twos_complement(read_cur_bytes) * 0.00269)  # Nm
        
        return torques

    def set_torque(self, torques):
        max_torque = 1.5  # Nm

        if len(torques) != len(self.motors):
            print("Wrong number of torques provided")
            return

        for i, (motor, torque) in enumerate(zip(self.motors, torques)):
            if torque > max_torque:
                torque = max_torque
                print(f"Torque too high for motor {i}, setting to max")
            elif torque < -max_torque:
                torque = -max_torque
                print(f"Torque too low for motor {i}, setting to min")

            current = int(torque / 0.00269)
            motor.write(XM430W210.GoalCurrent, current)


    def set_position(self, positions):

        if len(positions) != len(self.motors):
            print("Wrong number of positions provided")
            return
        
        for i, (motor, position) in enumerate(zip(self.motors, positions)):
            pos = int(((position + np.pi) * 4096 / (2 * np.pi)) % 4096)
            motor.write(XM430W210.GoalPosition, pos)

    def set_Kp(self, Kps):
        
        if len(Kps) != len(self.motors):
            print("Wrong number of Kps provided")
            return

        data = [int(Kp) for Kp in Kps]
        fields = [XM430W210.PositionPGain]

        self.u2d2.bulk_write(self.motor_ids, fields, data)
     
    
    def set_Kd(self, Kds):
        if len(Kds) != len(self.motors):
            print("Wrong number of Kds provided")
            return

        data = [int(Kd) for Kd in Kds]
        fields = [XM430W210.PositionDGain]

        self.u2d2.bulk_write(self.motor_ids, fields, data)
    
    def set_Ki(self, Kis):
        if len(Kis) != len(self.motors):
            print("Wrong number of Kis provided")
            return

        data = [int(Ki) for Ki in Kis]
        fields = [XM430W210.PositionIGain]

        self.u2d2.bulk_write(self.motor_ids, fields, data)

    def reset_gains(self):
        self.set_Kp([self.DEFAULT_KP] * len(self.motors))
        self.set_Kd([self.DEFAULT_KD] * len(self.motors))
        self.set_Ki([self.DEFAULT_KI] * len(self.motors))


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
