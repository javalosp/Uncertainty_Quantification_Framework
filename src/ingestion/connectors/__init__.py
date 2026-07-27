from .miso2_connector import MISO2Connector
from .iedc_connector import IEDCConnector
from .baci_connector import BACIConnector

# For additional data sources it is necessary to create the connector script
# and update this initialiser

__all__ = ['MISO2Connector', 'IEDCConnector', 'BACIConnector']