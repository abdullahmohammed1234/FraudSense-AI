"""
Multi-channel Payment Support Module.

This module provides fraud detection for various payment channels:
- Mobile payments (Apple Pay, Google Pay, Samsung Pay)
- Cryptocurrency transactions (BTC, ETH, etc.)
- Wire transfers (SWIFT, SEPA, domestic)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from datetime import datetime, timedelta


class PaymentChannel(Enum):
    """Payment channel types."""
    CARD = "card"
    MOBILE = "mobile"
    CRYPTO = "crypto"
    WIRE = "wire"


class MobileWallet(Enum):
    """Mobile wallet types."""
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    SAMSUNG_PAY = "samsung_pay"
    OTHER = "other"


class CryptoCurrency(Enum):
    """Supported cryptocurrencies."""
    BITCOIN = "btc"
    ETHEREUM = "eth"
    LITECOIN = "ltc"
    RIPPLE = "xrp"
    OTHER = "crypto"


class WireTransferType(Enum):
    """Wire transfer types."""
    SWIFT = "swift"
    SEPA = "sepa"
    CHIPS = "chips"
    DOMESTIC = "domestic"


@dataclass
class MultiChannelTransaction:
    """Transaction with multi-channel information."""
    transaction_id: str
    channel: PaymentChannel
    amount: float
    currency: str
    timestamp: datetime
    
    # Card-specific
    card_last_4: Optional[str] = None
    card_type: Optional[str] = None
    
    # Mobile-specific
    mobile_wallet: Optional[MobileWallet] = None
    device_id: Optional[str] = None
    device_os: Optional[str] = None
    
    # Crypto-specific
    crypto_currency: Optional[CryptoCurrency] = None
    wallet_address: Optional[str] = None
    blockchain_confirmations: Optional[int] = None
    
    # Wire-specific
    wire_type: Optional[WireTransferType] = None
    sender_bank: Optional[str] = None
    sender_country: Optional[str] = None
    recipient_country: Optional[str] = None
    
    # Common
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[str] = None


class MobilePaymentDetector:
    """Detector for mobile payment fraud."""
    
    RISK_WEIGHTS = {
        "new_device": 0.3,
        "unusual_location": 0.25,
        "multiple_wallets": 0.35,
        "high_value": 0.2,
        "unusual_time": 0.15,
    }
    
    def __init__(self):
        self.device_history: Dict[str, List[str]] = {}
        self.wallet_history: Dict[str, List[MobileWallet]] = {}
        self.location_history: Dict[str, List[str]] = {}
    
    def analyze_mobile_payment(
        self,
        transaction: MultiChannelTransaction,
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze mobile payment for fraud indicators."""
        risk_score = 0.0
        indicators = []
        
        user_id = transaction.user_id or "unknown"
        
        # Check device history
        if transaction.device_id:
            if user_id not in self.device_history:
                self.device_history[user_id] = []
            
            if transaction.device_id not in self.device_history[user_id]:
                risk_score += self.RISK_WEIGHTS["new_device"]
                indicators.append("New device detected")
                self.device_history[user_id].append(transaction.device_id)
        
        # Check wallet consistency
        if transaction.mobile_wallet and user_id in self.wallet_history:
            if transaction.mobile_wallet not in self.wallet_history[user_id]:
                risk_score += self.RISK_WEIGHTS["multiple_wallets"]
                indicators.append("Multiple wallet usage")
        
        # Store wallet
        if transaction.mobile_wallet:
            if user_id not in self.wallet_history:
                self.wallet_history[user_id] = []
            self.wallet_history[user_id].append(transaction.mobile_wallet)
        
        # Check location
        if transaction.location:
            if user_id not in self.location_history:
                self.location_history[user_id] = []
            
            if transaction.location not in self.location_history[user_id]:
                risk_score += self.RISK_WEIGHTS["unusual_location"]
                indicators.append("Unusual location")
                self.location_history[user_id].append(transaction.location)
        
        # Check amount
        if transaction.amount > 1000:
            risk_score += self.RISK_WEIGHTS["high_value"]
            if transaction.amount > 5000:
                indicators.append("Very high value transaction")
        
        # Check time patterns
        hour = transaction.timestamp.hour
        if hour < 5 or hour > 22:
            risk_score += self.RISK_WEIGHTS["unusual_time"]
            indicators.append("Unusual transaction time")
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        return {
            "channel_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "recommendation": "block" if risk_score > 0.6 else "review" if risk_score > 0.3 else "allow",
            "confidence": 0.85 if len(indicators) > 2 else 0.7
        }


class CryptoPaymentDetector:
    """Detector for cryptocurrency payment fraud."""
    
    RISK_WEIGHTS = {
        "new_wallet": 0.35,
        "low_confirmations": 0.4,
        "high_value_crypto": 0.25,
        "mixer_detected": 0.5,
        "cross_border": 0.2,
    }
    
    CONFIRMATION_THRESHOLDS = {
        "btc": 6,
        "eth": 12,
        "ltc": 12,
        "xrp": 1,
    }
    
    def __init__(self):
        self.wallet_blacklist: set = set()
        self.transaction_graph: Dict[str, List[str]] = {}
    
    def analyze_crypto_payment(
        self,
        transaction: MultiChannelTransaction,
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze cryptocurrency transaction for fraud indicators."""
        risk_score = 0.0
        indicators = []
        
        # Check wallet blacklist
        if transaction.wallet_address:
            if self._is_blacklisted(transaction.wallet_address):
                risk_score += 1.0
                indicators.append("Blacklisted wallet address")
        
        # Check confirmation threshold
        if transaction.crypto_currency and transaction.blockchain_confirmations:
            currency = transaction.crypto_currency.value
            threshold = self.CONFIRMATION_THRESHOLDS.get(currency, 6)
            
            if transaction.blockchain_confirmations < threshold:
                risk_score += self.RISK_WEIGHTS["low_confirmations"]
                indicators.append(f"Low blockchain confirmations: {transaction.blockchain_confirmations}")
        
        # Check for mixing patterns (simple heuristic)
        if transaction.wallet_address and self._has_mixer_pattern(transaction.wallet_address):
            risk_score += self.RISK_WEIGHTS["mixer_detected"]
            indicators.append("Potential mixer/rumble service detected")
        
        # Check high value
        if transaction.amount > 10000:
            risk_score += self.RISK_WEIGHTS["high_value_crypto"]
            indicators.append("High value crypto transaction")
        
        # Cross-border checks
        if transaction.sender_country and transaction.recipient_country:
            if transaction.sender_country != transaction.recipient_country:
                risk_score += self.RISK_WEIGHTS["cross_border"]
                indicators.append("Cross-border crypto transfer")
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "channel_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "recommendation": "block" if risk_score > 0.6 else "review" if risk_score > 0.3 else "allow",
            "confidence": 0.9,
            "confirmations_adequate": transaction.blockchain_confirmations >= self.CONFIRMATION_THRESHOLDS.get(
                transaction.crypto_currency.value if transaction.crypto_currency else "", 6
            )
        }
    
    def _is_blacklisted(self, wallet_address: str) -> bool:
        """Check if wallet is blacklisted."""
        return wallet_address.lower() in self.wallet_blacklist
    
    def _has_mixer_pattern(self, wallet_address: str) -> bool:
        """Heuristic check for mixer patterns."""
        return len(wallet_address) > 50


class WireTransferDetector:
    """Detector for wire transfer fraud."""
    
    RISK_WEIGHTS = {
        "new_beneficiary": 0.3,
        "high_risk_country": 0.4,
        "unusual_amount": 0.25,
        "complex_routing": 0.35,
        "rapid_sequence": 0.45,
    }
    
    HIGH_RISK_COUNTRIES = {
        "NK", "IR", "SY", "CU",  # Sanctioned
        "MM", "VE", "ZW",  # High risk
    }
    
    SUSPICIOUS_AMOUNTS = {
        "just_under_reporting": 9500,  # Just under $10k reporting threshold
        "round_amounts": 10000,
        "structured": 9000,
    }
    
    def __init__(self):
        self.beneficiary_history: Dict[str, List[str]] = {}
        self.sequence_tracker: Dict[str, List[datetime]] = {}
    
    def analyze_wire_transfer(
        self,
        transaction: MultiChannelTransaction,
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze wire transfer for fraud indicators."""
        risk_score = 0.0
        indicators = []
        
        user_id = transaction.user_id or "unknown"
        
        # Check beneficiary
        if transaction.recipient_country:
            if user_id not in self.beneficiary_history:
                self.beneficiary_history[user_id] = []
            
            if transaction.recipient_country not in self.beneficiary_history[user_id]:
                risk_score += self.RISK_WEIGHTS["new_beneficiary"]
                indicators.append("New beneficiary country")
                self.beneficiary_history[user_id].append(transaction.recipient_country)
        
        # Check high risk countries
        if transaction.recipient_country in self.HIGH_RISK_COUNTRIES:
            risk_score += self.RISK_WEIGHTS["high_risk_country"]
            indicators.append(f"High risk destination: {transaction.recipient_country}")
        
        # Check amount patterns
        if transaction.amount >= self.SUSPICIOUS_AMOUNTS["just_under_reporting"]:
            risk_score += self.RISK_WEIGHTS["unusual_amount"]
            indicators.append("Amount just under reporting threshold")
        
        if transaction.amount in [10000, 20000, 50000]:
            risk_score += self.RISK_WEIGHTS["unusual_amount"]
            indicators.append("Suspicious round amount")
        
        # Check rapid sequence
        if user_id in self.sequence_tracker:
            recent = [t for t in self.sequence_tracker[user_id] 
                     if (transaction.timestamp - t).total_seconds() < 3600]
            if len(recent) > 2:
                risk_score += self.RISK_WEIGHTS["rapid_sequence"]
                indicators.append("Rapid wire transfer sequence")
        
        # Store timestamp
        if user_id not in self.sequence_tracker:
            self.sequence_tracker[user_id] = []
        self.sequence_tracker[user_id].append(transaction.timestamp)
        
        risk_score = min(risk_score, 1.0)
        
        return {
            "channel_risk_score": round(risk_score, 4),
            "fraud_indicators": indicators,
            "recommendation": "block" if risk_score > 0.6 else "review" if risk_score > 0.3 else "allow",
            "confidence": 0.8,
            "reportable": transaction.amount >= 10000,
            "sanction_screened": transaction.recipient_country not in self.HIGH_RISK_COUNTRIES
        }


class MultiChannelFraudDetector:
    """Main detector for multi-channel fraud detection."""
    
    def __init__(self):
        self.mobile_detector = MobilePaymentDetector()
        self.crypto_detector = CryptoPaymentDetector()
        self.wire_detector = WireTransferDetector()
    
    def analyze_transaction(
        self,
        transaction: MultiChannelTransaction,
        user_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze transaction based on payment channel."""
        
        channel = transaction.channel
        
        if channel == PaymentChannel.MOBILE:
            return self.mobile_detector.analyze_mobile_payment(transaction, user_history)
        elif channel == PaymentChannel.CRYPTO:
            return self.crypto_detector.analyze_crypto_payment(transaction, user_history)
        elif channel == PaymentChannel.WIRE:
            return self.wire_detector.analyze_wire_transfer(transaction, user_history)
        else:
            return {
                "channel_risk_score": 0.0,
                "fraud_indicators": [],
                "recommendation": "allow",
                "confidence": 1.0
            }
    
    def get_channel_risk_config(self) -> Dict[str, Any]:
        """Get risk configuration for each channel."""
        return {
            "mobile": {
                "risk_weights": self.mobile_detector.RISK_WEIGHTS,
                "thresholds": {"block": 0.6, "review": 0.3}
            },
            "crypto": {
                "risk_weights": self.crypto_detector.RISK_WEIGHTS,
                "thresholds": {"block": 0.6, "review": 0.3}
            },
            "wire": {
                "risk_weights": self.wire_detector.RISK_WEIGHTS,
                "thresholds": {"block": 0.6, "review": 0.3}
            }
        }


_global_detector: Optional[MultiChannelFraudDetector] = None


def get_multi_channel_detector() -> MultiChannelFraudDetector:
    """Get global multi-channel detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = MultiChannelFraudDetector()
    return _global_detector


def create_transaction_from_features(features: Dict[str, Any]) -> MultiChannelTransaction:
    """Create MultiChannelTransaction from API features."""
    
    channel = features.get("payment_channel", "card")
    channel_map = {
        "mobile": PaymentChannel.MOBILE,
        "crypto": PaymentChannel.CRYPTO,
        "wire": PaymentChannel.WIRE,
        "card": PaymentChannel.CARD
    }
    
    # Parse mobile wallet
    wallet_str = features.get("mobile_wallet", "other")
    wallet_map = {
        "apple_pay": MobileWallet.APPLE_PAY,
        "google_pay": MobileWallet.GOOGLE_PAY,
        "samsung_pay": MobileWallet.SAMSUNG_PAY
    }
    
    # Parse crypto currency
    crypto_str = features.get("crypto_currency", "btc")
    crypto_map = {
        "btc": CryptoCurrency.BITCOIN,
        "eth": CryptoCurrency.ETHEREUM,
        "ltc": CryptoCurrency.LITECOIN,
        "xrp": CryptoCurrency.RIPPLE
    }
    
    # Parse wire type
    wire_str = features.get("wire_type", "domestic")
    wire_map = {
        "swift": WireTransferType.SWIFT,
        "sepa": WireTransferType.SEPA,
        "chips": WireTransferType.CHIPS,
        "domestic": WireTransferType.DOMESTIC
    }
    
    return MultiChannelTransaction(
        transaction_id=features.get("transaction_id", ""),
        channel=channel_map.get(channel, PaymentChannel.CARD),
        amount=features.get("Amount", 0.0),
        currency=features.get("currency", "USD"),
        timestamp=datetime.now(),
        card_last_4=features.get("card_last_4"),
        card_type=features.get("card_type"),
        mobile_wallet=wallet_map.get(wallet_str),
        device_id=features.get("device_id"),
        device_os=features.get("device_os"),
        crypto_currency=crypto_map.get(crypto_str),
        wallet_address=features.get("wallet_address"),
        blockchain_confirmations=features.get("blockchain_confirmations"),
        wire_type=wire_map.get(wire_str),
        sender_bank=features.get("sender_bank"),
        sender_country=features.get("sender_country"),
        recipient_country=features.get("recipient_country"),
        user_id=features.get("user_id"),
        ip_address=features.get("ip_address"),
        location=features.get("location")
    )