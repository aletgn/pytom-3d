"""
    fancy_log is part of pyCloM and it used to print nice log lines in the
    terminal in order to keep track of each manipulation step.
    
    Copyright (C) 2022  Alessandro Tognan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np

def milestone(message, log_level):
    if log_level == 0:
        pass
    else:
        sep = '*'
        no  = 80
        #
        if len(message) > no-4:
            print((sep*5) + ' ' + message + ' ' + (sep*5))
        else:
            message = list(message)
            header       = [sep]*no
            # below_header = [' ']*no
            # below_header[0] = sep
            # below_header[len(below_header)-1] = sep
            line   = [' ']*no
            line[0] = sep
            line[len(line)-1] = sep
            margin = round((len(header)-len(message))/2)
            #
            for h in range(0,len(message)):
                line[h+margin] = message[h]
            #
            print(''.join(header))
            # print(''.join(below_header))
            print(''.join(line))
            # print(''.join(below_header))
            print(''.join(header))

def single_operation(message,log_level,*value):
    if log_level == 0:
        pass
    else:
        if value:
            if isinstance(value[0], np.ndarray):
                print(f'----- {message:s}: {value[0]}')
            else:
                print(f'----- {message:s}: {value[0]:<.4f}')
        else:
            print(f'----- {message:s}')
