import logging
from typing import Dict, List, Any

class RelayController:
    """متحكم الريلاي - للتحكم في الأجهزة الطرفية"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.relays = self._initialize_relays()
        
    def _initialize_relays(self) -> Dict[str, Any]:
        """تهيئة الريلايات"""
        relays = {
            1: {'name': 'Emergency Cooling', 'pin': 23, 'state': False},
            2: {'name': 'Pressure Release', 'pin': 24, 'state': False},
            3: {'name': 'Gas Venting', 'pin': 25, 'state': False},
            4: {'name': 'Main Shutdown', 'pin': 26, 'state': False}
        }
        return relays
    
    def control_relay(self, relay_id: int, state: bool):
        """التحكم في حالة الريلاي"""
        try:
            if relay_id in self.relays:
                self.relays[relay_id]['state'] = state
                action = "activated" if state else "deactivated"
                self.logger.info(f"✅ Relay {relay_id} ({self.relays[relay_id]['name']}) {action}")
            else:
                self.logger.error(f"❌ Relay {relay_id} not found")
                
        except Exception as e:
            self.logger.error(f"❌ Error controlling relay {relay_id}: {e}")
    
    def get_relay_status(self) -> Dict[int, bool]:
        """الحصول على حالة جميع الريلايات"""
        return {rid: relay['state'] for rid, relay in self.relays.items()}
    
    def safe_shutdown(self):
        """إيقاف آمن لجميع الريلايات"""
        for relay_id in self.relays:
            self.control_relay(relay_id, False)
        self.logger.info("✅ All relays safely deactivated")
