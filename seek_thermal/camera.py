import os
import ctypes
import subprocess
from enum import IntEnum
import numpy as np
import cv2
from PIL import Image

class RetCode(IntEnum):
    SW_RETCODE_NONE = 0            #///< No error has been detected
    SW_RETCODE_NOTOPENED = 1       #///< Device is not opened
    SW_RETCODE_OPENEX = 2          #///< Device is already opened exclusively
    SW_RETCODE_BPARAM = 3          #///< Bad parameter
    SW_RETCODE_NOFRAME = 4         #///< Frame processing error
    SW_RETCODE_ERROR = 5           #///< Generic error
    SW_RETCODE_OFLOW = 6           #///< Some kind of overflow
    SW_RETCODE_USBERR = 7          #///< USB had an error; we need to restart the camera
    SW_RETCODE_SETONLY = 8         #///< Setting is write only
    SW_RETCODE_GETONLY = 9         #///< Setting is read only
    SW_RETCODE_NOTSUPPORTED = 10   #///< Setting is not supported by version of camera firmware
    SW_RETCODE_INVALIDSETUP = 11   #///< Another setting needs to be configured differently
    SW_RETCODE_DISCONNECTED = 12   #///< Device is disconnected from the host
    SW_RETCODE_LAST = 5000         #///< OEM error codes are >= 1000


# Defines the list of LUTs that can be applied to display video frames.
class DisplayLut(IntEnum):
    SW_LUT_FIRST = 1
    SW_LUT_WHITE = 1
    SW_LUT_BLACK = 17
    SW_LUT_IRON = 20
    SW_LUT_COOL = 2
    SW_LUT_AMBER = 9
    SW_LUT_INDIGO = 10
    SW_LUT_TYRIAN = 16
    SW_LUT_GLORY = 8
    SW_LUT_ENVY = 16
    SW_LUT_WHITE_NEW = 100
    SW_LUT_BLACK_NEW = 101
    SW_LUT_SPECTRA = 102
    SW_LUT_PRISM = 103
    SW_LUT_TYRIAN_NEW = 104
    SW_LUT_AMBER_NEW = 105
    SW_LUT_IRON_NEW = 106
    SW_LUT_HI = 107
    SW_LUT_HILO = 108
    SW_LUT_USER0 = 40000
    SW_LUT_USER1 = 40001
    SW_LUT_USER2 = 40002
    SW_LUT_USER3 = 40003
    SW_LUT_USER4 = 40004
    SW_LUT_LAST = SW_LUT_USER4


class DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_uint16),
        ("serialNumber", ctypes.c_char * 13),
        ("modelNumber", ctypes.c_char * 17),
        ("manufactureDate", ctypes.c_char * 33),
        ("fw_version_major", ctypes.c_uint8),
        ("fw_version_minor", ctypes.c_uint8),
        ("fw_build_major", ctypes.c_uint8),
        ("fw_build_minor", ctypes.c_uint8),
        ("frame_rows", ctypes.c_uint16),
        ("frame_cols", ctypes.c_uint16),
        ("usb_dev_handle", ctypes.c_void_p),
        ("usb_status", ctypes.c_void_p),
        ("ret_code", ctypes.c_int),
        ("seekware_context", ctypes.c_void_p), #/// Private SDK Context
        ("rawframe_rows", ctypes.c_uint16),
        ("rawframe_cols", ctypes.c_uint16),
        ("usb_device", ctypes.c_void_p),
    ]


class SdkInfo(ctypes.Structure):
    _fields_ = [
        ("sdk_version_major", ctypes.c_uint8),
        ("sdk_version_minor", ctypes.c_uint8),
        ("sdk_build_major", ctypes.c_uint8),
        ("sdk_build_minor", ctypes.c_uint8),
        ("lib_version_major", ctypes.c_uint8),
        ("lib_version_minor", ctypes.c_uint8),
        ("lib_build_major", ctypes.c_uint8),
        ("lib_build_minor", ctypes.c_uint8),
    ]

class Settings(IntEnum):                          #//  Size in Bytes:  Type:                           SET/GET:                            Description:
    SETTING_ACTIVE_LUT = 0                        #//  4               int                             SET/GET                             display lut used to fill display buffer in GetFrame/(Ex)
    SETTING_TEMP_UNITS = 1                        #//  4               int                             SET/GET                             temperature units
    SETTING_TIMEOUT = 2                           #//  4               int                             SET/GET                             USB timeout
    SETTING_CONTROL = 3                           #//  4               int                             SET/GET                             binary settings (legacy)
    SETTING_EMISSIVITY = 4                        #//  4               int                             SET/GET                             emissivity
    SETTING_BACKGROUND = 5                        #//  4               int                             SET/GET                             background temperature
    SETTING_THERMOGRAPHY_VERSION = 6              #//  4               int                             GET                                 thermography version
    SETTING_TEMP_DIODE_ROOM = 7                   #//  4               float                           GET                                 factory temperature used for estimating environment temp
    SETTING_TEMP_DIODE_SLOPE = 8                  #//  4               float                           GET                                 slope of FPA therm diode
    SETTING_TEMP_DIODE_OFFSET = 9                 #//  4               float                           GET                                 FPA therm diode offset
    SETTING_GLOBAL_THERM_ADJUST = 10              #//  4               sw_global_therm_adjust_t        SET                                 global temperature offset
    SETTING_SCENE_THERM_ADJUST = 11               #//  8               sw_scene_therm_adjust_t         SET                                 temperature offset for a specific scene
    SETTING_ENVIRONMENT_THERM_ADJUST = 12         #//  8               sw_environment_therm_adjust_t   SET                                 temperature offset for a specific environment
    SETTING_SPECIFIC_THERM_ADJUST = 13            #//  12              sw_specific_therm_adjust_t      SET                                 temperature offset for a specific scene and environment
    SETTING_TRANSIENT_CORRECTION_ENABLE = 14      #//  4               uint32_t                        SET/GET                             transient correction
    SETTING_TRANSIENT_CORRECTION_PARAMS = 15      #//  8               sw_transient_adjust_t           SET/GET                             amplitude and decay for transient correction
    SETTING_SMOOTHING = 16                        #//  4               uint32_t                        SET/GET                             image smoothing
    SETTING_AUTOSHUTTER = 17                      #//  4               uint32_t                        SET/GET                             autoshutter
    SETTING_MINMAX = 18                           #//  24              sw_minmax_t                     GET                                 MIN/MAX with coordinates
    SETTING_SHARPENING = 19                       #//  4               uint32_t                        SET/GET                             image sharpening
    SETTING_ENABLE_TIMESTAMP = 20                 #//  4               uint32_t                        SET/GET                             timestamping
    SETTING_RESET_TIMESTAMP = 21                  #//  4               uint32_t                        SET                                 reset timestamping counter
    SETTING_TRIGGER_SHUTTER = 22                  #//  4               uint32_t                        SET                                 shutter trigger

    #// AGC Settings
    SETTING_AGC_MODE = 23                         #//  4               uint32_t                        SET/GET                             AGC Mode (0=Legacy HistEQ, 1=Linear Min/Max, 2=HistEQ)
    SETTING_HISTEQ_BIN_COUNT = 24                 #//  4               uint32_t                        GET                                 number of bins
    SETTING_HISTEQ_INPUT_BIT_DEPTH = 25           #//  4               uint32_t                        GET                                 number of input bits before performing AGC calculation
    SETTING_HISTEQ_OUTPUT_BIT_DEPTH = 26          #//  4               uint32_t                        GET                                 number of output bits after performing AGC calculation
    SETTING_HISTEQ_HIST_WIDTH_COUNTS = 27         #//  4               uint32_t                        GET                                 width of the active histogram
    SETTING_HISTEQ_PLATEAU_VALUE = 28             #//  4               float                           SET/GET                             max percentage of total pixels that can be assigned to a single bin
    SETTING_HISTEQ_GAIN_LIMIT = 29                #//  4               float                           SET/GET                             max percentage of output colors per sensor count
    SETTING_HISTEQ_GAIN_LIMIT_FACTOR_ENABLE = 30  #//  4               uint32_t                        SET/GET                             gain limit factor
    SETTING_HISTEQ_GAIN_LIMIT_FACTOR = 31         #//  4               float                           GET                                 gain limit factor value
    SETTING_HISTEQ_GAIN_LIMIT_FACTOR_XMAX = 32    #//  4               uint32_t                        SET/GET                             max histogram width where gain limit factor will affect the total gain
    SETTING_HISTEQ_GAIN_LIMIT_FACTOR_YMIN = 33    #//  4               float                           SET/GET                             min gain limit factor value
    SETTING_HISTEQ_ALPHA_TIME = 34                #//  4               float                           SET/GET                             number of seconds to blend current frame's histogram with previous frame's histogram
    SETTING_HISTEQ_TRIM_LEFT = 35                 #//  4               float                           SET/GET                             percentage of outliers to trim from left side of histogram
    SETTING_HISTEQ_TRIM_RIGHT = 36                #//  4               float                           SET/GET                             percentage of outliers to trim from right side of histogram
    SETTING_LINMINMAX_MODE = 37                   #//  4               uint32_t                        SET/GET                             linear min/max mode (0=Min and Max Auto, 1=Min and Max Locked, 2=Min Locked and Max Auto, 3=Min Auto and Max Locked)
    SETTING_LINMINMAX_MIN_LOCK = 38               #//  4               uint32_t                        SET/GET                             lower bound for linear min/max (only used when SETTING_LINMINMAX_MODE = 1 or 2)
    SETTING_LINMINMAX_MAX_LOCK = 39               #//  4               uint32_t                        SET/GET                             upper bound for linear min/max (only used when SETTING_LINMINMAX_MODE = 1 or 3)
    SETTING_LINMINMAX_ACTIVE_MIN_VALUE = 40       #//  4               uint32_t                        GET                                 min sensor count value in the last scene
    SETTING_LINMINMAX_ACTIVE_MAX_VALUE = 41       #//  4               uint32_t                        GET                                 max sensor count value in the last scene

    FEATURE_OEM = 1000 #//  Size in bytes and type dependent on setting. Please contact Seek for details.       Use FEATURE_OEM + index with additional settings provided by Seek
    SETTING_LAST = 100000


class Camera():

    def __init__(self):
        self.__camera = None
        self.__st = None
    
    def seek_lib(self):
        paths = os.getenv('LD_LIBRARY_PATH')
        if not paths:
            raise Exception('Environment variable LD_LIBRARY_PATH not set.') 

        for path in paths.split(':'):
            p = os.path.join(path, 'libseekware.so')
            if os.path.exists(p):
                return p
            
        raise Exception('Could not find libseekware.so within any directory of LD_LIBRARY_PATH.') 
    
    def open(self):
        libpath = self.seek_lib()
        print(f'Loading {libpath}')
        self.__st = ctypes.CDLL(libpath)
        info = SdkInfo()
        _ = self.__st.Seekware_GetSdkInfo(None, ctypes.byref(info))
        print('==== Seek Thermal SDK info ====')
        for field_name, field_type in info._fields_:
            print(field_name, getattr(info, field_name))
        NUM_CAMS = ctypes.c_int(1)
        num_cameras_found = ctypes.c_int(0)
        CameraListArray = DeviceInfo * NUM_CAMS.value
        camera_list = ctypes.pointer(CameraListArray())
        _ = self.__st.Seekware_Find(ctypes.pointer(camera_list), NUM_CAMS, ctypes.pointer(num_cameras_found))
        if num_cameras_found.value == 0:
            raise Exception('Could not find any Seek Thermal cameras.')
        self.__camera = camera_list[0][0]
        ret = self.__st.Seekware_Open(ctypes.pointer(self.__camera))
        if ret != RetCode.SW_RETCODE_NONE:
            raise Exception('Failure opening camera({ret}).')
        print('==== Camera info ====')
        for field_name, field_type in self.__camera._fields_:
            print(field_name, getattr(self.__camera, field_name))
        ret = self.__st.Seekware_SetSettingEx(ctypes.pointer(self.__camera),
                                       ctypes.c_uint32(int(Settings.SETTING_ACTIVE_LUT)),
                                       ctypes.byref(ctypes.c_uint32(int(DisplayLut.SW_LUT_TYRIAN_NEW))),
                                       ctypes.c_uint32(4))
        if ret != RetCode.SW_RETCODE_NONE:
            raise Exception('Failure setting default LUT({ret}).')
        ret = self.__st.Seekware_SetSettingEx(ctypes.pointer(self.__camera),
                                       ctypes.c_uint32(int(Settings.SETTING_AUTOSHUTTER)),
                                       ctypes.byref(ctypes.c_uint32(1)),
                                       ctypes.c_uint32(4))
        if ret != RetCode.SW_RETCODE_NONE:
            raise Exception('Failure setting auto-shutter({ret}).')

    def close(self):
        if self.__st and self.__camera:
            ret = self.__st.Seekware_Close(ctypes.pointer(self.__camera))
            self.__st = None
            self.__camera = None
            if ret != RetCode.SW_RETCODE_NONE:
                raise Exception('Failure closing the camera({ret}).')

    def read_images(self):
        if self.__st and self.__camera:
            frame_pixels = self.__camera.frame_cols * self.__camera.frame_rows
            TemperatureArray = ctypes.c_float * frame_pixels
            DisplayArray = ctypes.c_uint8 * frame_pixels * 4
            FilteredArray = ctypes.c_uint16 * frame_pixels
            temperature = TemperatureArray()
            display = DisplayArray()
            filtered = FilteredArray()
            ret = self.__st.Seekware_GetImage(ctypes.pointer(self.__camera),
                                       ctypes.pointer(filtered),
                                       ctypes.pointer(temperature),
                                       ctypes.pointer(display))
            if ret != RetCode.SW_RETCODE_NONE:
                raise Exception('Failure retrieving images({ret}).')
            temperatures_np = np.ctypeslib.as_array(temperature)
            temperatures_np = temperatures_np.reshape((self.__camera.frame_rows,
                                                       self.__camera.frame_cols))
            filtered_np = np.ctypeslib.as_array(filtered)
            filtered_np = filtered_np.reshape((self.__camera.frame_rows,
                self.__camera.frame_cols))
            
            display_np = np.ctypeslib.as_array(display)
            display_np = display_np.reshape((self.__camera.frame_rows,
                                             self.__camera.frame_cols, 4))
            return temperatures_np, display_np, filtered_np
        else:
            return None, None


if __name__ == "__main__":
    CAMERA_RESOLUTION_X = 640
    CAMERA_RESOLUTION_Y = 480
    CAMERA_FPS = 8
    CROP_MARGIN = abs(round((CAMERA_RESOLUTION_X-CAMERA_RESOLUTION_Y)/2.0))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Video capture could not be started')

    command = "v4l2-ctl --set-ctrl=scene_mode=11"
    _ = subprocess.call(command, shell=True)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_X)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_Y)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    sk = Camera()
    sk.open()
    temperatures, image, filtered = sk.read_images()

    ret, cv2_image = cap.read()
    if not ret:
        print(f'Failed to acquire image({ret})')
    else:
        cv2_image = cv2.flip(cv2_image, 0)
        cv2_image = cv2.flip(cv2_image, 1)

        #--- make image square to avoid different x, y scaling
        #Getting the bigger side of the image
        s = max(cv2_image.shape[0:2])
        #Creating a dark square with NUMPY  
        f = np.zeros((s,s,3),np.uint8)
        #Getting the centering position
        ax,ay = (s - cv2_image.shape[1])//2,(s - cv2_image.shape[0])//2
        #Pasting the 'image' in a centering position
        f[ay:cv2_image.shape[0]+ay,ax:ax+cv2_image.shape[1]] = cv2_image
        cv2_image = f
        #cv2_image = cv2.cvtColor(f, cv2.COLOR_BGRA2RGB)
        cv2.imwrite("visible.jpg", cv2_image)
        cv2.imshow("Visible", cv2_image)

    np.savetxt("thermography.csv", temperatures, delimiter=",", fmt="%.2f")            
    cv2.imwrite("display.jpg", image)
    cv2.imshow("Display", image)
    im = Image.fromarray(filtered)
    im.save('filtered.tif')
    sk.close()

    gray = filtered
    #gray = cv2.imread('filtered.tif',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Normalized", normalized)
    cv2.imwrite("normalized.jpg", normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imshow("Final", color)

    cv2.waitKey(0)

