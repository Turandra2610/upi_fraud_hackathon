from openenv.core.env_server import create_fastapi_app
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import UPIAction, UPIObservation
from server.upi_project_environment import UPIFraudEnvironment

env = UPIFraudEnvironment()
app = create_fastapi_app(env, UPIAction, UPIObservation)
