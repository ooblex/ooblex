"""
OpenTelemetry distributed tracing configuration
Shared across all services
"""
import os
from typing import Optional, Dict, Any
import logging

from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode
from opentelemetry.metrics import get_meter_provider, set_meter_provider

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for OpenTelemetry tracing"""
    
    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        environment: str = "development",
        additional_attributes: Optional[Dict[str, Any]] = None
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTLP_ENDPOINT", "localhost:4317")
        self.environment = environment
        self.additional_attributes = additional_attributes or {}
        
        # Create resource
        self.resource = Resource.create({
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "service.environment": environment,
            "service.instance.id": os.getenv("HOSTNAME", "local"),
            **self.additional_attributes
        })
        
        # Initialize providers
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
    
    def setup_tracing(self, enable_console_export: bool = False):
        """Setup distributed tracing"""
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=self.resource)
        
        # Add OTLP exporter
        if self.otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True  # Use TLS in production
                )
                self.tracer_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
                logger.info(f"OTLP tracing exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup OTLP exporter: {e}")
        
        # Add console exporter for debugging
        if enable_console_export:
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
        
        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Set propagator for distributed tracing
        set_global_textmap(B3MultiFormat())
        
        # Get tracer
        self.tracer = trace.get_tracer(
            self.service_name,
            self.service_version
        )
        
        logger.info(f"Tracing initialized for {self.service_name}")
    
    def setup_metrics(self, prometheus_port: int = 9090):
        """Setup metrics collection"""
        # Create meter provider with Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        
        self.meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[prometheus_reader]
        )
        
        # Add OTLP metrics exporter
        if self.otlp_endpoint:
            try:
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True
                )
                # Note: In production, add proper metric reader with OTLP exporter
                logger.info(f"OTLP metrics exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup OTLP metrics exporter: {e}")
        
        # Set global meter provider
        set_meter_provider(self.meter_provider)
        
        # Get meter
        self.meter = metrics.get_meter(
            self.service_name,
            self.service_version
        )
        
        logger.info(f"Metrics initialized for {self.service_name}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application"""
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=self.tracer_provider,
            excluded_urls="health,metrics,docs,openapi.json"
        )
        logger.info("FastAPI instrumentation enabled")
    
    def instrument_grpc_server(self):
        """Instrument gRPC server"""
        GrpcInstrumentorServer().instrument(
            tracer_provider=self.tracer_provider
        )
        logger.info("gRPC server instrumentation enabled")
    
    def instrument_grpc_client(self):
        """Instrument gRPC client"""
        GrpcInstrumentorClient().instrument(
            tracer_provider=self.tracer_provider
        )
        logger.info("gRPC client instrumentation enabled")
    
    def instrument_redis(self):
        """Instrument Redis client"""
        RedisInstrumentor().instrument(
            tracer_provider=self.tracer_provider
        )
        logger.info("Redis instrumentation enabled")
    
    def instrument_asyncpg(self):
        """Instrument AsyncPG"""
        AsyncPGInstrumentor().instrument(
            tracer_provider=self.tracer_provider
        )
        logger.info("AsyncPG instrumentation enabled")
    
    def instrument_aiohttp(self):
        """Instrument aiohttp client"""
        AioHttpClientInstrumentor().instrument(
            tracer_provider=self.tracer_provider
        )
        logger.info("Aiohttp instrumentation enabled")
    
    def instrument_logging(self):
        """Instrument logging to include trace context"""
        LoggingInstrumentor().instrument(
            tracer_provider=self.tracer_provider,
            set_logging_format=True
        )
        logger.info("Logging instrumentation enabled")


# Decorators for tracing
def trace_method(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace a method"""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        async def async_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    span.set_attributes(attributes)
                
                # Add function arguments as attributes
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise
        
        def sync_wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    span.set_attributes(attributes)
                
                # Add function arguments as attributes
                span.set_attribute("function.args", str(args))
                span.set_attribute("function.kwargs", str(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context propagation helpers
def inject_trace_context(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject trace context into headers"""
    from opentelemetry.propagate import inject
    inject(headers)
    return headers


def extract_trace_context(headers: Dict[str, str]):
    """Extract trace context from headers"""
    from opentelemetry.propagate import extract
    return extract(headers)


# Metrics helpers
class MetricsCollector:
    """Helper class for collecting custom metrics"""
    
    def __init__(self, meter):
        self.meter = meter
        self._counters = {}
        self._histograms = {}
        self._gauges = {}
    
    def get_counter(self, name: str, description: str = "", unit: str = "1"):
        """Get or create a counter"""
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._counters[name]
    
    def get_histogram(self, name: str, description: str = "", unit: str = "ms"):
        """Get or create a histogram"""
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._histograms[name]
    
    def get_gauge(self, name: str, description: str = "", unit: str = "1"):
        """Get or create an observable gauge"""
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_observable_gauge(
                name=name,
                description=description,
                unit=unit
            )
        return self._gauges[name]
    
    def record_duration(self, name: str, duration_ms: float, attributes: Optional[Dict[str, Any]] = None):
        """Record a duration metric"""
        histogram = self.get_histogram(f"{name}_duration", f"Duration of {name}", "ms")
        histogram.record(duration_ms, attributes=attributes)
    
    def increment_counter(self, name: str, value: int = 1, attributes: Optional[Dict[str, Any]] = None):
        """Increment a counter"""
        counter = self.get_counter(f"{name}_total", f"Total count of {name}")
        counter.add(value, attributes=attributes)


# Example usage for services
def setup_service_tracing(service_name: str, enable_console: bool = False) -> TracingConfig:
    """Setup tracing for a service"""
    config = TracingConfig(
        service_name=service_name,
        service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
        otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
        environment=os.getenv("ENVIRONMENT", "development")
    )
    
    # Setup tracing and metrics
    config.setup_tracing(enable_console_export=enable_console)
    config.setup_metrics()
    
    # Instrument libraries
    config.instrument_redis()
    config.instrument_aiohttp()
    config.instrument_logging()
    
    return config