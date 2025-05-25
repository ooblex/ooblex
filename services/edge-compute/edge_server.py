import asyncio
import aiohttp
from aiohttp import web
import aioredis
import json
import hashlib
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeComputeServer:
    def __init__(self, redis_url='redis://localhost', port=8090):
        self.redis_url = redis_url
        self.port = port
        self.redis = None
        self.app = web.Application()
        self.setup_routes()
        
        # WASM module management
        self.wasm_modules = {
            'face_detection': {
                'path': '../../html/wasm/face_detection.js',
                'version': '1.0.0',
                'size_limit': 512 * 1024,  # 512KB
                'capabilities': ['face_detection', 'face_tracking']
            },
            'style_transfer_lite': {
                'path': '../../html/wasm/style_transfer_lite.js',
                'version': '1.0.0',
                'size_limit': 1024 * 1024,  # 1MB
                'capabilities': ['sketch', 'oil_painting', 'watercolor']
            },
            'background_blur': {
                'path': '../../html/wasm/background_blur.js',
                'version': '1.0.0',
                'size_limit': 512 * 1024,
                'capabilities': ['blur', 'bokeh', 'depth_based_blur']
            }
        }
        
        # Edge worker management
        self.edge_workers = {}
        self.worker_capabilities = {}
        self.worker_performance = {}
        
        # Model distribution
        self.model_cache = {}
        self.model_versions = {}
        
    def setup_routes(self):
        """Setup API routes"""
        # WASM module endpoints
        self.app.router.add_get('/api/edge/modules', self.list_modules)
        self.app.router.add_get('/api/edge/modules/{module_name}', self.get_module)
        self.app.router.add_get('/api/edge/modules/{module_name}/version', self.get_module_version)
        
        # Worker management
        self.app.router.add_post('/api/edge/workers/register', self.register_worker)
        self.app.router.add_post('/api/edge/workers/{worker_id}/heartbeat', self.worker_heartbeat)
        self.app.router.add_post('/api/edge/workers/{worker_id}/report', self.worker_report)
        self.app.router.add_get('/api/edge/workers/status', self.get_workers_status)
        
        # Model distribution
        self.app.router.add_get('/api/edge/models', self.list_models)
        self.app.router.add_get('/api/edge/models/{model_name}', self.get_model)
        self.app.router.add_post('/api/edge/models/{model_name}/deploy', self.deploy_model)
        
        # Processing coordination
        self.app.router.add_post('/api/edge/process', self.coordinate_processing)
        self.app.router.add_get('/api/edge/capabilities', self.get_edge_capabilities)
        
        # Health and metrics
        self.app.router.add_get('/api/edge/health', self.health_check)
        self.app.router.add_get('/api/edge/metrics', self.get_metrics)
        
    async def start(self):
        """Start the edge compute server"""
        try:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
            logger.info(f"Connected to Redis at {self.redis_url}")
            
            # Start background tasks
            asyncio.create_task(self.cleanup_inactive_workers())
            asyncio.create_task(self.monitor_performance())
            
            # Start web server
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logger.info(f"Edge compute server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start edge compute server: {e}")
            raise
    
    async def list_modules(self, request):
        """List available WASM modules"""
        modules = []
        for name, info in self.wasm_modules.items():
            modules.append({
                'name': name,
                'version': info['version'],
                'size_limit': info['size_limit'],
                'capabilities': info['capabilities']
            })
        
        return web.json_response({
            'modules': modules,
            'total': len(modules)
        })
    
    async def get_module(self, request):
        """Serve WASM module"""
        module_name = request.match_info['module_name']
        
        if module_name not in self.wasm_modules:
            return web.json_response(
                {'error': f'Module {module_name} not found'},
                status=404
            )
        
        module_info = self.wasm_modules[module_name]
        module_path = os.path.join(os.path.dirname(__file__), module_info['path'])
        
        if not os.path.exists(module_path):
            # Build module if not exists
            await self.build_wasm_module(module_name)
        
        try:
            with open(module_path, 'rb') as f:
                content = f.read()
            
            # Add caching headers
            etag = hashlib.md5(content).hexdigest()
            headers = {
                'Content-Type': 'application/javascript',
                'ETag': etag,
                'Cache-Control': 'public, max-age=3600',
                'X-Module-Version': module_info['version']
            }
            
            # Check if client has cached version
            if request.headers.get('If-None-Match') == etag:
                return web.Response(status=304)
            
            return web.Response(body=content, headers=headers)
            
        except Exception as e:
            logger.error(f"Failed to serve module {module_name}: {e}")
            return web.json_response(
                {'error': 'Failed to serve module'},
                status=500
            )
    
    async def get_module_version(self, request):
        """Get module version info"""
        module_name = request.match_info['module_name']
        
        if module_name not in self.wasm_modules:
            return web.json_response(
                {'error': f'Module {module_name} not found'},
                status=404
            )
        
        module_info = self.wasm_modules[module_name]
        
        return web.json_response({
            'module': module_name,
            'version': module_info['version'],
            'capabilities': module_info['capabilities'],
            'size_limit': module_info['size_limit']
        })
    
    async def register_worker(self, request):
        """Register an edge worker"""
        try:
            data = await request.json()
            worker_id = data.get('worker_id')
            capabilities = data.get('capabilities', [])
            device_info = data.get('device_info', {})
            
            if not worker_id:
                return web.json_response(
                    {'error': 'worker_id required'},
                    status=400
                )
            
            # Store worker info
            worker_info = {
                'id': worker_id,
                'capabilities': capabilities,
                'device_info': device_info,
                'registered_at': datetime.utcnow().isoformat(),
                'last_heartbeat': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            self.edge_workers[worker_id] = worker_info
            self.worker_capabilities[worker_id] = set(capabilities)
            self.worker_performance[worker_id] = {
                'processing_times': [],
                'success_rate': 1.0,
                'load': 0.0
            }
            
            # Store in Redis
            await self.redis.setex(
                f'edge:worker:{worker_id}',
                3600,  # 1 hour TTL
                json.dumps(worker_info)
            )
            
            logger.info(f"Registered edge worker: {worker_id}")
            
            return web.json_response({
                'status': 'registered',
                'worker_id': worker_id,
                'assigned_models': self._get_assigned_models(capabilities)
            })
            
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return web.json_response(
                {'error': 'Registration failed'},
                status=500
            )
    
    async def worker_heartbeat(self, request):
        """Handle worker heartbeat"""
        worker_id = request.match_info['worker_id']
        
        if worker_id not in self.edge_workers:
            return web.json_response(
                {'error': 'Worker not found'},
                status=404
            )
        
        try:
            data = await request.json()
            
            # Update worker status
            self.edge_workers[worker_id]['last_heartbeat'] = datetime.utcnow().isoformat()
            self.edge_workers[worker_id]['status'] = 'active'
            
            # Update performance metrics
            if 'metrics' in data:
                metrics = data['metrics']
                perf = self.worker_performance[worker_id]
                perf['load'] = metrics.get('load', 0.0)
                
                if 'processing_time' in metrics:
                    perf['processing_times'].append(metrics['processing_time'])
                    # Keep only last 100 measurements
                    perf['processing_times'] = perf['processing_times'][-100:]
            
            # Store in Redis
            await self.redis.setex(
                f'edge:worker:{worker_id}',
                3600,
                json.dumps(self.edge_workers[worker_id])
            )
            
            return web.json_response({
                'status': 'ok',
                'next_heartbeat': 30  # seconds
            })
            
        except Exception as e:
            logger.error(f"Failed to process heartbeat: {e}")
            return web.json_response(
                {'error': 'Heartbeat processing failed'},
                status=500
            )
    
    async def worker_report(self, request):
        """Handle worker performance report"""
        worker_id = request.match_info['worker_id']
        
        if worker_id not in self.edge_workers:
            return web.json_response(
                {'error': 'Worker not found'},
                status=404
            )
        
        try:
            data = await request.json()
            
            # Update performance metrics
            perf = self.worker_performance[worker_id]
            
            if 'success_rate' in data:
                perf['success_rate'] = data['success_rate']
            
            if 'processing_stats' in data:
                stats = data['processing_stats']
                await self.redis.hset(
                    'edge:stats',
                    worker_id,
                    json.dumps(stats)
                )
            
            return web.json_response({'status': 'reported'})
            
        except Exception as e:
            logger.error(f"Failed to process worker report: {e}")
            return web.json_response(
                {'error': 'Report processing failed'},
                status=500
            )
    
    async def get_workers_status(self, request):
        """Get status of all edge workers"""
        workers = []
        
        for worker_id, info in self.edge_workers.items():
            perf = self.worker_performance.get(worker_id, {})
            avg_processing_time = (
                sum(perf.get('processing_times', [])) / len(perf.get('processing_times', [1]))
                if perf.get('processing_times') else 0
            )
            
            workers.append({
                'id': worker_id,
                'status': info['status'],
                'capabilities': info['capabilities'],
                'last_heartbeat': info['last_heartbeat'],
                'performance': {
                    'avg_processing_time': avg_processing_time,
                    'success_rate': perf.get('success_rate', 1.0),
                    'current_load': perf.get('load', 0.0)
                }
            })
        
        return web.json_response({
            'workers': workers,
            'total': len(workers),
            'active': sum(1 for w in workers if w['status'] == 'active')
        })
    
    async def coordinate_processing(self, request):
        """Coordinate processing between edge and cloud"""
        try:
            data = await request.json()
            task_type = data.get('type')
            priority = data.get('priority', 'normal')
            
            # Find capable workers
            capable_workers = self._find_capable_workers(task_type)
            
            if not capable_workers:
                # Fallback to cloud processing
                return web.json_response({
                    'mode': 'cloud',
                    'reason': 'no_capable_edge_workers',
                    'endpoint': '/api/process'
                })
            
            # Select best worker based on load and performance
            selected_worker = self._select_best_worker(capable_workers)
            
            if not selected_worker:
                return web.json_response({
                    'mode': 'cloud',
                    'reason': 'edge_workers_overloaded',
                    'endpoint': '/api/process'
                })
            
            # Return edge processing instructions
            return web.json_response({
                'mode': 'edge',
                'worker_id': selected_worker,
                'module': self._get_module_for_task(task_type),
                'fallback': {
                    'mode': 'cloud',
                    'endpoint': '/api/process',
                    'timeout': 5000  # ms
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to coordinate processing: {e}")
            return web.json_response(
                {'error': 'Coordination failed'},
                status=500
            )
    
    async def get_edge_capabilities(self, request):
        """Get current edge computing capabilities"""
        capabilities = set()
        total_capacity = 0.0
        
        for worker_id, caps in self.worker_capabilities.items():
            if self.edge_workers[worker_id]['status'] == 'active':
                capabilities.update(caps)
                perf = self.worker_performance[worker_id]
                total_capacity += (1.0 - perf.get('load', 0.0))
        
        return web.json_response({
            'capabilities': list(capabilities),
            'active_workers': sum(1 for w in self.edge_workers.values() 
                                if w['status'] == 'active'),
            'total_capacity': total_capacity,
            'modules': list(self.wasm_modules.keys())
        })
    
    async def list_models(self, request):
        """List available models for edge deployment"""
        models = [
            {
                'name': 'face_detection_lite',
                'type': 'detection',
                'size': 2048,  # KB
                'format': 'tflite',
                'version': '1.0.0'
            },
            {
                'name': 'style_transfer_mobile',
                'type': 'style_transfer',
                'size': 4096,
                'format': 'tflite',
                'version': '1.0.0'
            },
            {
                'name': 'segmentation_lite',
                'type': 'segmentation',
                'size': 3072,
                'format': 'tflite',
                'version': '1.0.0'
            }
        ]
        
        return web.json_response({
            'models': models,
            'total': len(models)
        })
    
    async def get_model(self, request):
        """Get model for edge deployment"""
        model_name = request.match_info['model_name']
        
        # Simulate model serving
        # In production, this would serve actual model files
        
        return web.json_response({
            'model': model_name,
            'url': f'/api/edge/models/{model_name}/download',
            'metadata': {
                'input_shape': [1, 224, 224, 3],
                'output_shape': [1, 1000],
                'preprocessing': 'normalize'
            }
        })
    
    async def deploy_model(self, request):
        """Deploy model to edge workers"""
        model_name = request.match_info['model_name']
        
        try:
            data = await request.json()
            target_workers = data.get('workers', [])
            
            if not target_workers:
                # Deploy to all capable workers
                target_workers = list(self.edge_workers.keys())
            
            # Simulate deployment
            deployment_status = {}
            for worker_id in target_workers:
                if worker_id in self.edge_workers:
                    deployment_status[worker_id] = 'deployed'
                    # Store deployment info
                    await self.redis.sadd(
                        f'edge:model:{model_name}:workers',
                        worker_id
                    )
            
            return web.json_response({
                'model': model_name,
                'deployment_status': deployment_status,
                'deployed_count': len(deployment_status)
            })
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return web.json_response(
                {'error': 'Deployment failed'},
                status=500
            )
    
    async def health_check(self, request):
        """Health check endpoint"""
        redis_status = 'connected' if self.redis else 'disconnected'
        
        return web.json_response({
            'status': 'healthy',
            'redis': redis_status,
            'active_workers': sum(1 for w in self.edge_workers.values() 
                                if w['status'] == 'active'),
            'modules_available': len(self.wasm_modules)
        })
    
    async def get_metrics(self, request):
        """Get edge compute metrics"""
        metrics = {
            'workers': {
                'total': len(self.edge_workers),
                'active': sum(1 for w in self.edge_workers.values() 
                            if w['status'] == 'active'),
                'inactive': sum(1 for w in self.edge_workers.values() 
                              if w['status'] != 'active')
            },
            'processing': {
                'edge_requests': await self._get_metric('edge_requests'),
                'cloud_fallbacks': await self._get_metric('cloud_fallbacks'),
                'avg_edge_latency': await self._get_metric('avg_edge_latency'),
                'avg_cloud_latency': await self._get_metric('avg_cloud_latency')
            },
            'models': {
                'deployed': await self._get_deployed_models_count(),
                'cache_hit_rate': await self._get_metric('model_cache_hit_rate')
            }
        }
        
        return web.json_response(metrics)
    
    async def cleanup_inactive_workers(self):
        """Clean up inactive workers periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                inactive_threshold = timedelta(minutes=5)
                
                for worker_id, info in list(self.edge_workers.items()):
                    last_heartbeat = datetime.fromisoformat(info['last_heartbeat'])
                    
                    if current_time - last_heartbeat > inactive_threshold:
                        logger.info(f"Removing inactive worker: {worker_id}")
                        del self.edge_workers[worker_id]
                        del self.worker_capabilities[worker_id]
                        del self.worker_performance[worker_id]
                        await self.redis.delete(f'edge:worker:{worker_id}')
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def monitor_performance(self):
        """Monitor edge computing performance"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Calculate aggregate metrics
                total_load = 0
                total_workers = 0
                avg_processing_times = []
                
                for worker_id, perf in self.worker_performance.items():
                    if self.edge_workers[worker_id]['status'] == 'active':
                        total_load += perf.get('load', 0)
                        total_workers += 1
                        
                        if perf.get('processing_times'):
                            avg_time = sum(perf['processing_times']) / len(perf['processing_times'])
                            avg_processing_times.append(avg_time)
                
                if total_workers > 0:
                    avg_load = total_load / total_workers
                    overall_avg_time = (
                        sum(avg_processing_times) / len(avg_processing_times)
                        if avg_processing_times else 0
                    )
                    
                    # Store metrics
                    await self.redis.hset(
                        'edge:metrics',
                        'avg_load',
                        str(avg_load)
                    )
                    await self.redis.hset(
                        'edge:metrics',
                        'avg_processing_time',
                        str(overall_avg_time)
                    )
                    
                    logger.info(f"Edge compute metrics - Load: {avg_load:.2f}, "
                              f"Avg time: {overall_avg_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
    
    async def build_wasm_module(self, module_name):
        """Build WASM module if needed"""
        logger.info(f"Building WASM module: {module_name}")
        
        # Run make command to build module
        import subprocess
        
        wasm_dir = os.path.join(os.path.dirname(__file__), 'wasm_modules')
        result = subprocess.run(
            ['make', module_name],
            cwd=wasm_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to build module: {result.stderr}")
            raise Exception(f"Module build failed: {module_name}")
        
        logger.info(f"Successfully built module: {module_name}")
    
    def _find_capable_workers(self, task_type):
        """Find workers capable of handling task type"""
        capable = []
        
        for worker_id, capabilities in self.worker_capabilities.items():
            if (task_type in capabilities and 
                self.edge_workers[worker_id]['status'] == 'active'):
                capable.append(worker_id)
        
        return capable
    
    def _select_best_worker(self, worker_ids):
        """Select best worker based on load and performance"""
        best_worker = None
        best_score = -1
        
        for worker_id in worker_ids:
            perf = self.worker_performance[worker_id]
            load = perf.get('load', 0)
            success_rate = perf.get('success_rate', 1)
            
            # Simple scoring: lower load and higher success rate is better
            score = (1 - load) * success_rate
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        # Only select if score is above threshold
        return best_worker if best_score > 0.3 else None
    
    def _get_module_for_task(self, task_type):
        """Get appropriate WASM module for task type"""
        task_module_map = {
            'face_detection': 'face_detection',
            'face_tracking': 'face_detection',
            'style_transfer': 'style_transfer_lite',
            'sketch': 'style_transfer_lite',
            'background_blur': 'background_blur',
            'bokeh': 'background_blur'
        }
        
        return task_module_map.get(task_type, 'face_detection')
    
    def _get_assigned_models(self, capabilities):
        """Get models assigned based on capabilities"""
        model_capability_map = {
            'face_detection': ['face_detection_lite'],
            'style_transfer': ['style_transfer_mobile'],
            'segmentation': ['segmentation_lite']
        }
        
        assigned = []
        for cap in capabilities:
            if cap in model_capability_map:
                assigned.extend(model_capability_map[cap])
        
        return list(set(assigned))
    
    async def _get_metric(self, metric_name):
        """Get metric from Redis"""
        try:
            value = await self.redis.hget('edge:metrics', metric_name)
            return float(value) if value else 0.0
        except:
            return 0.0
    
    async def _get_deployed_models_count(self):
        """Get count of deployed models"""
        try:
            count = 0
            cursor = b'0'
            while cursor:
                cursor, keys = await self.redis.scan(
                    cursor, 
                    match='edge:model:*:workers'
                )
                count += len(keys)
                if cursor == b'0':
                    break
            return count
        except:
            return 0


async def main():
    """Main entry point"""
    server = EdgeComputeServer()
    await server.start()
    
    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down edge compute server")
    finally:
        if server.redis:
            server.redis.close()
            await server.redis.wait_closed()


if __name__ == '__main__':
    asyncio.run(main()