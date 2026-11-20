"""W8A8 entry point; use the same arguments as convert_int8.py."""
import sys
from convert_int8 import main
if __name__ == '__main__':
    sys.argv += ['--family', 'yolo26']
    main()
