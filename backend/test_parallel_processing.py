#!/usr/bin/env python3
"""
Test script for parallel document processing with Redis/Celery.
"""

import asyncio
import httpx
import time
import json
from pathlib import Path
import os
from typing import List, Dict, Any


class ParallelProcessingTester:
    """Test harness for parallel document processing."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def create_test_files(self, num_files: int = 5, size_mb: int = 10) -> List[Path]:
        """Create test files of specified size."""
        test_dir = Path("test_documents")
        test_dir.mkdir(exist_ok=True)
        
        files = []
        content_size = size_mb * 1024 * 1024  # Convert to bytes
        
        for i in range(num_files):
            file_path = test_dir / f"test_doc_{i+1}.txt"
            
            # Create file with dummy content
            with open(file_path, 'w') as f:
                # Write content in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB chunks
                for _ in range(size_mb):
                    f.write("x" * chunk_size)
            
            files.append(file_path)
            print(f"Created test file: {file_path} ({size_mb}MB)")
        
        return files
    
    async def upload_files(self, files: List[Path], dataset_name: str) -> Dict[str, Any]:
        """Upload files to the API."""
        print(f"\nUploading {len(files)} files to dataset '{dataset_name}'...")
        
        file_handles = []
        for file_path in files:
            file_handles.append(
                ('files', (file_path.name, open(file_path, 'rb'), 'text/plain'))
            )
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/upload",
                files=file_handles,
                data={
                    "dataset_name": dataset_name,
                    "use_celery": "true"
                }
            )
            
            result = response.json()
            print(f"Upload response: {json.dumps(result, indent=2)}")
            return result
            
        finally:
            # Close file handles
            for _, (_, file_handle, _) in file_handles:
                file_handle.close()
    
    async def monitor_job(self, job_id: str, poll_interval: int = 2) -> Dict[str, Any]:
        """Monitor job progress until completion."""
        print(f"\nMonitoring job: {job_id}")
        
        start_time = time.time()
        last_progress = -1
        
        while True:
            response = await self.client.get(f"{self.base_url}/api/jobs/{job_id}/status")
            
            if response.status_code == 404:
                print(f"Job {job_id} not found")
                return None
            
            job_status = response.json()
            status = job_status.get("status", "unknown")
            progress = job_status.get("progress", 0)
            
            # Print progress if changed
            if progress != last_progress:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Status: {status}, Progress: {progress:.1f}%, "
                      f"Processed: {job_status.get('documents_processed', 0)}, "
                      f"Failed: {job_status.get('documents_failed', 0)}, "
                      f"Chunks: {job_status.get('chunks_created', 0)}")
                last_progress = progress
            
            # Check if job is complete
            if status in ["completed", "failed", "cancelled", "timeout"]:
                elapsed = time.time() - start_time
                print(f"\nJob completed in {elapsed:.1f} seconds")
                print(f"Final status: {json.dumps(job_status, indent=2)}")
                return job_status
            
            await asyncio.sleep(poll_interval)
    
    async def test_parallel_processing(
        self,
        num_files: int = 5,
        file_size_mb: int = 10,
        dataset_name: str = "test_parallel"
    ):
        """Run parallel processing test."""
        print(f"\n{'='*60}")
        print(f"Parallel Processing Test")
        print(f"Files: {num_files}, Size: {file_size_mb}MB each")
        print(f"Total data: {num_files * file_size_mb}MB")
        print(f"{'='*60}")
        
        # Create test files
        files = await self.create_test_files(num_files, file_size_mb)
        
        # Upload files
        upload_result = await self.upload_files(files, dataset_name)
        
        if upload_result.get("status") == "queued":
            job_id = upload_result.get("job_id")
            
            # Monitor job
            job_result = await self.monitor_job(job_id)
            
            if job_result:
                # Print summary
                print(f"\n{'='*60}")
                print("Test Summary:")
                print(f"- Total files: {num_files}")
                print(f"- File size: {file_size_mb}MB each")
                print(f"- Total data: {num_files * file_size_mb}MB")
                print(f"- Documents processed: {job_result.get('documents_processed', 0)}")
                print(f"- Documents failed: {job_result.get('documents_failed', 0)}")
                print(f"- Chunks created: {job_result.get('chunks_created', 0)}")
                print(f"- Status: {job_result.get('status')}")
                print(f"{'='*60}")
        
        # Cleanup test files
        for file_path in files:
            file_path.unlink(missing_ok=True)
        
        print("\nTest files cleaned up")
    
    async def test_large_file(
        self,
        file_size_mb: int = 100,
        dataset_name: str = "test_large"
    ):
        """Test processing of a single large file."""
        print(f"\n{'='*60}")
        print(f"Large File Processing Test")
        print(f"File size: {file_size_mb}MB")
        print(f"{'='*60}")
        
        # Create large test file
        files = await self.create_test_files(1, file_size_mb)
        
        # Upload file
        upload_result = await self.upload_files(files, dataset_name)
        
        if upload_result.get("status") == "queued":
            job_id = upload_result.get("job_id")
            
            # Monitor job with longer poll interval
            job_result = await self.monitor_job(job_id, poll_interval=5)
            
            if job_result:
                print(f"\n{'='*60}")
                print("Large File Test Summary:")
                print(f"- File size: {file_size_mb}MB")
                print(f"- Processing time: Check logs above")
                print(f"- Chunks created: {job_result.get('chunks_created', 0)}")
                print(f"- Status: {job_result.get('status')}")
                print(f"{'='*60}")
        
        # Cleanup
        for file_path in files:
            file_path.unlink(missing_ok=True)
        
        print("\nTest file cleaned up")
    
    async def get_job_metrics(self):
        """Get overall job metrics."""
        response = await self.client.get(f"{self.base_url}/api/jobs/metrics")
        metrics = response.json()
        
        print(f"\n{'='*60}")
        print("Job Processing Metrics:")
        print(f"- Total jobs: {metrics.get('total_jobs', 0)}")
        print(f"- Active jobs: {metrics.get('active_jobs', 0)}")
        print(f"- Status distribution: {metrics.get('status_distribution', {})}")
        print(f"{'='*60}")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()


async def main():
    """Run tests."""
    tester = ParallelProcessingTester()
    
    try:
        # Test 1: Multiple small files in parallel
        await tester.test_parallel_processing(
            num_files=5,
            file_size_mb=5,
            dataset_name="test_parallel_small"
        )
        
        await asyncio.sleep(2)
        
        # Test 2: Multiple medium files in parallel
        await tester.test_parallel_processing(
            num_files=3,
            file_size_mb=20,
            dataset_name="test_parallel_medium"
        )
        
        await asyncio.sleep(2)
        
        # Test 3: Single large file
        await tester.test_large_file(
            file_size_mb=100,
            dataset_name="test_large_file"
        )
        
        # Get metrics
        await tester.get_job_metrics()
        
    finally:
        await tester.cleanup()
        
        # Clean up test directory
        test_dir = Path("test_documents")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
            print("\nTest directory cleaned up")


if __name__ == "__main__":
    print("Starting Parallel Processing Tests...")
    print("Make sure the backend is running with Redis and Celery workers")
    print("-" * 60)
    
    asyncio.run(main())