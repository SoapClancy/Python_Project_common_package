import os
import re

python_project_common_path_ = os.path.dirname(os.path.abspath(__file__))
# python_project_common_path_ = os.path.abspath(os.path.join(python_project_common_path_, "../"))
python_project_common_path_ = python_project_common_path_ + '/'
python_project_common_path_ = re.sub(r'\\', '/', python_project_common_path_)
