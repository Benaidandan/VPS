import torch
from pathlib import Path
from typing import List, Dict, Union, Optional
import shutil
from hloc import extract_features, pairs_from_retrieval, extractors
from hloc.utils.base_model import dynamic_load
from hloc.extractors import megaloc
import time
class VisualPlaceRecognition:
    """Visual Place Recognition module using Hloc's methods."""
    
    def __init__(self, config: Dict):
        """
        Initialize the VPR module.
        
        Args:
            config: Configuration dictionary containing VPR settings
        """
        self.config = config
        self.method = config['vpr']['method']
        self.top_k = config['vpr']['top_k']
        # Set up the feature extractor configuration
        if self.method == 'netvlad':
            self.retrieval_conf = extract_features.confs['netvlad']
        elif self.method == 'megaloc':
            self.retrieval_conf = extract_features.confs['megaloc']
        else:
            raise ValueError(f"Unsupported VPR method: {self.method}")
        #megaloc vpr
        self.model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])(self.retrieval_conf["model"]).eval().to(config['system']['device'])
    def extract_global_descriptors(self, 
                                 images: Union[str, Path, List[Union[str, Path]]], 
                                 output_path: Union[str, Path]) -> Path:
        """
        Extract global descriptors from images.
        
        Args:
            images: Path to image or list of image paths
            output_path: Dath to save the descriptors
            
        Returns:
            Path to the saved descriptors
        """
        return extract_features.main(self.retrieval_conf, images,feature_path=output_path, overwrite=True, model=self.model)


    def find_similar_images(self,
                          query_image: Union[str, Path],) -> Path:
        """
        Complete pipeline to find similar images for a query image.
        
        Args:
            query_image: Path to the query image

            
        Returns:
            Path to the pairs file containing similar images
        """
        # Ensure ref_images is a list
        ref_images = Path(self.config['vpr']['ref_images_path']) / "rgb"
        if isinstance(ref_images, (str, Path)):
            ref_images = Path(ref_images)
        
        output_dir = Path(self.config['vpr']['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        
        # Create unified query image with fixed name (overwrite mode)
        unified_query_dir = output_dir / "temp"
        print("Cleaning up query directory...")
        if unified_query_dir.exists():
            shutil.rmtree(unified_query_dir)
            print(f"✓ Cleaned up: {unified_query_dir}")
        unified_query_dir.mkdir(exist_ok=True, parents=True)

        # 直接以原文件名复制到目标目录，会覆盖同名文件
        shutil.copy2(query_image, unified_query_dir / Path(query_image).name)
        
        # Check if reference descriptors already exist
        ref_descriptors_path = Path(self.config['vpr']['ref_descriptors_path'])
        
        # Extract reference descriptors only if they don't exist
        if not ref_descriptors_path.exists():
            print("Extracting reference descriptors...")
            self.extract_global_descriptors(ref_images, ref_descriptors_path)
        else:
            print("Using existing reference descriptors...")
        
        # Extract query descriptors (always overwrite due to unified naming)
        print("Extracting query descriptors...")
        start_time = time.time()
        query_descriptors = self.extract_global_descriptors(unified_query_dir, Path(self.config['vpr']['query_descriptors_path']))
        end_time = time.time()
        print(f"query_image: {Path(query_image).name} extract_global_descriptors time: {end_time - start_time:.2f} seconds")
        
        start_time = time.time()
        pairs_from_retrieval.main(
            query_descriptors,
            self.config['vpr']['pairs_file_path'],
            self.top_k,
            db_descriptors=ref_descriptors_path
        )
        end_time = time.time()
        print(f"query_image: {Path(query_image).name} pairs_from_retrieval time: {end_time - start_time:.2f} seconds")
        
        return self.config['vpr']['pairs_file_path']