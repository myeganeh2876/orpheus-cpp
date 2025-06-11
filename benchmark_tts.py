#!/usr/bin/env python3
"""
Comprehensive TTS Benchmarking Script for Dual RTX 6000 Ada Setup

This script benchmarks both the original and optimized versions of Orpheus CPP
on 100 paragraph texts and provides detailed performance metrics.
"""

import asyncio
import json
import time
import statistics
import psutil
import threading
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import GPUtil
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("GPUtil not available. Install with: pip install GPUtil")

try:
    from src.orpheus_cpp.model import OrpheusCpp
    from src.orpheus_cpp.optimized_model import OptimizedOrpheusCpp, TTSOptions
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    print("pip install cupy-cuda12x")
    sys.exit(1)


# Sample paragraph texts for benchmarking
BENCHMARK_TEXTS = [
    "The rapid advancement of artificial intelligence has transformed numerous industries, from healthcare to finance, creating unprecedented opportunities for innovation and efficiency. Machine learning algorithms now power everything from recommendation systems to autonomous vehicles, fundamentally changing how we interact with technology in our daily lives.",
    
    "Climate change represents one of the most pressing challenges of our time, requiring immediate and coordinated global action to reduce greenhouse gas emissions and transition to renewable energy sources. The consequences of inaction extend far beyond environmental concerns, affecting economic stability, food security, and human migration patterns worldwide.",
    
    "The digital revolution has democratized access to information and education, enabling people from all backgrounds to learn new skills and connect with others across the globe. Online platforms have created new forms of collaboration and knowledge sharing that were unimaginable just a few decades ago.",
    
    "Space exploration continues to capture human imagination and drive technological innovation, with private companies now joining government agencies in the quest to explore Mars and establish sustainable human presence beyond Earth. These endeavors push the boundaries of engineering and inspire future generations of scientists and explorers.",
    
    "The human brain remains one of the most complex and fascinating structures in the known universe, containing approximately 86 billion neurons that form intricate networks responsible for consciousness, memory, and creativity. Neuroscience research continues to unlock the mysteries of how our minds process information and generate thoughts.",
    
    "Renewable energy technologies have reached a tipping point where they are now cost-competitive with fossil fuels in many markets, accelerating the global transition to clean energy systems. Solar panels, wind turbines, and battery storage solutions are becoming increasingly efficient and affordable for both residential and commercial applications.",
    
    "The rise of remote work has fundamentally altered traditional employment patterns, offering greater flexibility for workers while challenging organizations to maintain productivity and company culture in distributed teams. This shift has implications for urban planning, real estate markets, and work-life balance across different industries.",
    
    "Biotechnology advances are revolutionizing medicine through personalized treatments, gene therapy, and precision diagnostics that target diseases at the molecular level. These innovations promise to extend human lifespan and improve quality of life for millions of people suffering from previously incurable conditions.",
    
    "The Internet of Things is creating a interconnected world where everyday objects can communicate and share data, enabling smart cities, efficient supply chains, and automated homes that respond to human needs. This connectivity brings both convenience and new challenges related to privacy and cybersecurity.",
    
    "Quantum computing represents a paradigm shift in computational power, with the potential to solve complex problems that are intractable for classical computers, including drug discovery, cryptography, and optimization challenges across various scientific and industrial domains.",
    
    "The global food system faces mounting pressure to feed a growing population while minimizing environmental impact, driving innovation in sustainable agriculture, vertical farming, and alternative protein sources. These developments are crucial for ensuring food security in an era of climate change and resource scarcity.",
    
    "Virtual and augmented reality technologies are transforming entertainment, education, and professional training by creating immersive experiences that blur the line between digital and physical worlds. These platforms offer new ways to visualize complex data, practice dangerous procedures, and connect with others in shared virtual spaces.",
    
    "The democratization of financial services through fintech innovations has expanded access to banking, lending, and investment opportunities for underserved populations worldwide. Mobile payment systems, cryptocurrency, and robo-advisors are reshaping how people manage and grow their wealth.",
    
    "Artificial neural networks inspired by biological brain structures have achieved remarkable breakthroughs in pattern recognition, natural language processing, and decision-making tasks that were once thought to require human-level intelligence. These systems continue to evolve and find applications in diverse fields.",
    
    "The circular economy model promotes sustainable resource use by designing products for reuse, recycling, and regeneration, challenging the traditional linear take-make-dispose approach that has dominated industrial production for centuries. This shift requires collaboration across entire supply chains and changes in consumer behavior.",
    
    "Gene editing technologies like CRISPR have opened new possibilities for treating genetic diseases, improving crop yields, and understanding fundamental biological processes. However, these powerful tools also raise important ethical questions about the limits and responsibilities of human intervention in natural systems.",
    
    "The sharing economy has created new business models that leverage underutilized assets and enable peer-to-peer transactions, from ride-sharing and home rentals to skill-sharing platforms. This trend reflects changing consumer preferences toward access over ownership and community-based solutions.",
    
    "Cybersecurity has become a critical concern as our dependence on digital systems grows, with threats ranging from individual identity theft to nation-state attacks on critical infrastructure. Protecting sensitive data and maintaining system integrity requires constant vigilance and adaptive security measures.",
    
    "The aging global population presents both challenges and opportunities, driving demand for healthcare services, assistive technologies, and age-friendly urban design while also creating a wealth of experience and knowledge that can benefit society. Addressing demographic shifts requires comprehensive policy responses and innovative solutions.",
    
    "Ocean conservation efforts are becoming increasingly urgent as marine ecosystems face threats from pollution, overfishing, and climate change. Protecting these vital environments requires international cooperation, sustainable fishing practices, and innovative technologies for monitoring and restoration.",
    
    "The evolution of transportation systems toward electric and autonomous vehicles promises to reduce emissions, improve safety, and transform urban mobility patterns. This transition requires significant infrastructure investments and coordination between public and private sectors to realize its full potential.",
    
    "Educational technology is personalizing learning experiences through adaptive algorithms, virtual classrooms, and interactive content that adjusts to individual student needs and learning styles. These tools have the potential to make high-quality education more accessible and effective for learners worldwide.",
    
    "The pharmaceutical industry is embracing artificial intelligence and machine learning to accelerate drug discovery, optimize clinical trials, and identify new therapeutic targets. These technologies can significantly reduce the time and cost required to bring life-saving medications to market.",
    
    "Smart manufacturing systems integrate sensors, robotics, and data analytics to create more efficient, flexible, and responsive production processes. Industry 4.0 technologies enable mass customization, predictive maintenance, and real-time optimization of manufacturing operations.",
    
    "The preservation of cultural heritage in the digital age involves using advanced technologies like 3D scanning, virtual reality, and blockchain to document, protect, and share historical artifacts and traditions with future generations. These efforts help maintain cultural diversity in an increasingly connected world.",
    
    "Precision agriculture uses satellite imagery, drones, and sensor networks to optimize crop management, reduce resource waste, and increase yields while minimizing environmental impact. These technologies enable farmers to make data-driven decisions about irrigation, fertilization, and pest control.",
    
    "The development of brain-computer interfaces opens new possibilities for treating neurological conditions, controlling prosthetic devices, and enhancing human cognitive abilities. These technologies represent a convergence of neuroscience, engineering, and computer science with profound implications for human potential.",
    
    "Sustainable urban planning integrates green infrastructure, public transportation, and mixed-use development to create livable cities that minimize environmental impact while supporting economic growth and social equity. These approaches are essential for accommodating growing urban populations.",
    
    "The field of synthetic biology combines engineering principles with biological systems to design and construct new biological parts, devices, and systems for useful purposes. This emerging discipline has applications in medicine, manufacturing, agriculture, and environmental remediation.",
    
    "Blockchain technology extends beyond cryptocurrency to enable secure, transparent, and decentralized systems for supply chain management, digital identity verification, and smart contracts. These applications have the potential to reduce fraud, increase efficiency, and create new forms of digital trust.",
    
    "The study of human longevity and aging processes is revealing new insights into the biological mechanisms that determine lifespan and healthspan. Research in this field may lead to interventions that extend healthy human life and reduce the burden of age-related diseases.",
    
    "Renewable energy storage solutions are critical for managing the intermittent nature of solar and wind power, enabling grid stability and energy security as we transition away from fossil fuels. Advanced battery technologies, pumped hydro storage, and other innovations are making clean energy more reliable.",
    
    "The integration of artificial intelligence in healthcare is improving diagnostic accuracy, treatment planning, and patient outcomes through analysis of medical images, electronic health records, and genomic data. These tools augment human expertise and enable more personalized medical care.",
    
    "Nanotechnology applications span from targeted drug delivery and water purification to advanced materials and electronics, operating at the scale of atoms and molecules to create products with enhanced properties and capabilities. This field continues to push the boundaries of what's possible in science and engineering.",
    
    "The concept of digital twins creates virtual replicas of physical systems, enabling real-time monitoring, simulation, and optimization of everything from individual machines to entire cities. This technology improves maintenance, reduces downtime, and supports better decision-making across industries.",
    
    "Marine biotechnology harnesses the unique properties of ocean organisms to develop new pharmaceuticals, biomaterials, and industrial processes. The vast biodiversity of marine environments offers untapped potential for scientific discovery and sustainable innovation.",
    
    "The transition to a hydrogen economy involves developing infrastructure for producing, storing, and distributing hydrogen as a clean energy carrier for transportation, industry, and power generation. This shift could play a crucial role in decarbonizing sectors that are difficult to electrify.",
    
    "Computational biology and bioinformatics are accelerating our understanding of complex biological systems through the analysis of large-scale genomic, proteomic, and metabolomic datasets. These fields enable personalized medicine, drug discovery, and insights into evolutionary processes.",
    
    "The development of autonomous systems extends beyond self-driving cars to include drones, robots, and other machines capable of operating independently in complex environments. These technologies have applications in logistics, agriculture, search and rescue, and space exploration.",
    
    "Social media platforms have fundamentally changed how people communicate, share information, and form communities, creating new opportunities for connection while also raising concerns about privacy, misinformation, and mental health. Understanding these impacts is crucial for navigating the digital age.",
    
    "The field of materials science is developing new substances with extraordinary properties, from superconductors and shape-memory alloys to self-healing materials and programmable matter. These innovations enable advances in electronics, aerospace, medicine, and countless other applications.",
    
    "Ecosystem restoration efforts are using scientific understanding of ecological processes to repair damaged environments, restore biodiversity, and enhance ecosystem services that support human well-being. These projects demonstrate the potential for healing our relationship with the natural world.",
    
    "The democratization of space access through reusable rockets and small satellites is enabling new scientific missions, commercial ventures, and global communication networks. This trend is making space-based services more affordable and accessible to a broader range of users.",
    
    "Personalized nutrition uses genetic information, microbiome analysis, and lifestyle data to provide individualized dietary recommendations that optimize health outcomes and prevent disease. This approach recognizes that nutritional needs vary significantly among individuals.",
    
    "The integration of renewable energy sources into existing power grids requires sophisticated management systems that can balance supply and demand in real-time while maintaining grid stability. Smart grid technologies enable this transition to a more sustainable energy system.",
    
    "Advances in 3D printing and additive manufacturing are enabling on-demand production of complex parts, customized products, and even living tissues. These technologies are transforming manufacturing, healthcare, aerospace, and construction industries by reducing waste and enabling new design possibilities.",
    
    "The study of extremophiles, organisms that thrive in extreme environments, is providing insights into the limits of life and potential for life on other planets. These discoveries inform astrobiology research and inspire biotechnology applications in harsh industrial conditions.",
    
    "Digital health technologies, including wearable devices, mobile health apps, and telemedicine platforms, are empowering individuals to monitor their health, manage chronic conditions, and access medical care remotely. These tools are particularly valuable for underserved populations and rural communities.",
    
    "The circular design philosophy emphasizes creating products that can be easily disassembled, repaired, and recycled at the end of their useful life. This approach reduces waste, conserves resources, and creates new business opportunities in the growing circular economy.",
    
    "Cognitive computing systems that can understand, reason, and learn from data are augmenting human decision-making in complex domains like finance, healthcare, and scientific research. These systems combine artificial intelligence with human expertise to solve challenging problems.",
    
    "The development of lab-grown meat and other cellular agriculture technologies offers the potential to produce animal proteins without the environmental impact and ethical concerns associated with traditional livestock farming. These innovations could transform the global food system.",
    
    "Quantum sensors and metrology devices leverage quantum mechanical properties to achieve unprecedented precision in measuring time, gravity, magnetic fields, and other physical quantities. These instruments enable new scientific discoveries and technological applications.",
    
    "The field of synthetic chemistry is developing new methods for creating complex molecules with desired properties, enabling the production of pharmaceuticals, materials, and chemicals that would be difficult or impossible to obtain from natural sources.",
    
    "Urban vertical farming systems use controlled environments and hydroponic or aeroponic growing methods to produce fresh vegetables and herbs in city centers, reducing transportation costs and environmental impact while providing year-round local food production.",
    
    "The integration of artificial intelligence in creative industries is generating new forms of art, music, and literature while raising questions about authorship, creativity, and the role of human artists in an age of machine-generated content.",
    
    "Bioengineering approaches to environmental remediation use living organisms to clean up pollution, restore contaminated sites, and remove harmful substances from air, water, and soil. These biological solutions offer sustainable alternatives to traditional cleanup methods.",
    
    "The development of neuromorphic computing architectures that mimic the structure and function of biological neural networks promises to create more efficient and adaptive computing systems for artificial intelligence applications.",
    
    "Precision medicine approaches use molecular profiling, genetic testing, and other advanced diagnostics to tailor treatments to individual patients, improving efficacy while reducing side effects and healthcare costs.",
    
    "The emergence of digital currencies and central bank digital currencies is reshaping monetary systems and payment infrastructure, offering new possibilities for financial inclusion while raising questions about privacy and monetary policy.",
    
    "Advanced robotics systems are becoming more capable of working alongside humans in collaborative environments, from manufacturing floors to healthcare settings, augmenting human capabilities rather than simply replacing human workers.",
    
    "The study of the human microbiome is revealing the crucial role that microbial communities play in health and disease, leading to new therapeutic approaches that target these complex ecosystems rather than individual pathogens.",
    
    "Sustainable aviation fuels and electric aircraft technologies are being developed to reduce the carbon footprint of air travel, which is essential for meeting global climate goals while maintaining connectivity in an interconnected world.",
    
    "The application of machine learning to drug discovery is accelerating the identification of new therapeutic compounds and optimizing their properties, potentially reducing the time and cost required to develop life-saving medications.",
    
    "Smart city initiatives integrate sensors, data analytics, and citizen engagement platforms to improve urban services, reduce resource consumption, and enhance quality of life for residents while addressing challenges like traffic congestion and air pollution.",
    
    "The development of biodegradable plastics and alternative packaging materials is addressing the global plastic pollution crisis by creating products that break down naturally in the environment without leaving harmful residues.",
    
    "Advances in gene therapy are enabling the treatment of previously incurable genetic diseases by correcting defective genes or introducing new genetic material to fight disease, offering hope to patients with rare and common conditions alike.",
    
    "The field of computational social science uses big data and advanced analytics to understand human behavior, social networks, and collective phenomena, providing insights that inform policy decisions and social interventions.",
    
    "Renewable energy microgrids enable communities to generate, store, and distribute clean energy locally, increasing energy security and resilience while reducing dependence on centralized power systems and fossil fuels.",
    
    "The integration of virtual and augmented reality in healthcare is improving medical training, patient education, and therapeutic interventions by creating immersive experiences that enhance understanding and engagement.",
    
    "Advances in water purification and desalination technologies are addressing global water scarcity by making it more affordable and energy-efficient to produce clean drinking water from contaminated or saline sources.",
    
    "The development of autonomous underwater vehicles and marine robotics is enabling new forms of ocean exploration, environmental monitoring, and underwater construction that were previously impossible or extremely dangerous for human divers.",
    
    "Personalized learning platforms use artificial intelligence to adapt educational content and pacing to individual student needs, learning styles, and progress, potentially improving educational outcomes and reducing achievement gaps.",
    
    "The field of tissue engineering combines cells, biomaterials, and growth factors to create functional tissues and organs for transplantation, drug testing, and disease modeling, offering new hope for patients with organ failure.",
    
    "Smart agriculture systems integrate Internet of Things sensors, satellite imagery, and predictive analytics to optimize crop management, reduce resource waste, and increase yields while adapting to changing climate conditions.",
    
    "The development of carbon capture and utilization technologies is creating new ways to remove carbon dioxide from the atmosphere and convert it into useful products, potentially helping to mitigate climate change while creating economic value.",
    
    "Advances in battery technology are enabling longer-lasting, faster-charging, and more sustainable energy storage solutions that support the transition to electric vehicles and renewable energy systems.",
    
    "The application of blockchain technology to supply chain management is increasing transparency, traceability, and accountability in global trade, helping to combat counterfeiting and ensure ethical sourcing of products.",
    
    "Computational fluid dynamics and advanced simulation tools are enabling engineers to design more efficient aircraft, vehicles, and industrial systems by modeling complex flow patterns and optimizing performance before physical prototypes are built.",
    
    "The study of epigenetics is revealing how environmental factors can influence gene expression without changing DNA sequences, providing new insights into disease development and potential therapeutic targets.",
    
    "Digital fabrication technologies, including 3D printing, laser cutting, and computer-controlled machining, are democratizing manufacturing by enabling individuals and small businesses to produce custom products and prototypes.",
    
    "The development of smart textiles and wearable electronics is creating new possibilities for health monitoring, human-computer interaction, and adaptive clothing that responds to environmental conditions and user needs.",
    
    "Advances in renewable energy forecasting use machine learning and weather data to predict solar and wind power generation, enabling better grid management and integration of variable renewable energy sources.",
    
    "The field of astrobiology combines astronomy, biology, and geology to search for life beyond Earth and understand the conditions that enable life to emerge and evolve in the universe.",
    
    "Precision agriculture drones equipped with multispectral cameras and sensors can monitor crop health, detect pest infestations, and optimize irrigation and fertilization with unprecedented accuracy and efficiency.",
    
    "The development of solid-state batteries promises to deliver higher energy density, improved safety, and longer lifespan compared to conventional lithium-ion batteries, potentially revolutionizing electric vehicles and energy storage.",
    
    "Biometric authentication technologies are evolving beyond fingerprints and facial recognition to include behavioral patterns, voice recognition, and even DNA analysis, providing more secure and convenient identity verification methods.",
    
    "The integration of artificial intelligence in financial services is improving fraud detection, risk assessment, and algorithmic trading while enabling new forms of personalized financial advice and automated investment management.",
    
    "Advanced materials research is developing substances with programmable properties that can change shape, stiffness, or other characteristics in response to external stimuli, opening new possibilities for adaptive structures and devices.",
    
    "The field of digital pathology uses high-resolution imaging and artificial intelligence to analyze tissue samples and medical images, potentially improving diagnostic accuracy and enabling remote consultation between specialists.",
    
    "Sustainable packaging innovations are creating biodegradable, compostable, and reusable alternatives to traditional packaging materials, helping to reduce waste and environmental impact throughout the supply chain.",
    
    "The development of quantum communication networks promises ultra-secure data transmission using the principles of quantum mechanics, potentially revolutionizing cybersecurity and enabling new forms of distributed computing.",
    
    "Advances in stem cell research are opening new possibilities for regenerative medicine, disease modeling, and drug discovery by harnessing the ability of these cells to differentiate into various tissue types.",
    
    "The application of machine learning to climate modeling is improving our ability to predict weather patterns, understand climate change impacts, and develop more effective mitigation and adaptation strategies.",
    
    "Smart building technologies integrate sensors, automation systems, and data analytics to optimize energy consumption, improve occupant comfort, and reduce maintenance costs while enhancing safety and security.",
    
    "The development of lab-on-a-chip devices is miniaturizing complex laboratory procedures onto small, portable platforms that can perform rapid diagnostic tests and chemical analyses in point-of-care settings.",
    
    "Advances in computational linguistics and natural language processing are enabling more sophisticated human-computer interaction, automated translation, and text analysis capabilities that bridge language barriers and extract insights from vast amounts of textual data.",
    
    "The field of synthetic biology is engineering biological systems to produce pharmaceuticals, biofuels, and other valuable compounds, potentially creating more sustainable and efficient manufacturing processes.",
    
    "Digital twin technology is being applied to entire cities, creating virtual models that can simulate traffic patterns, energy consumption, and urban development scenarios to support better planning and management decisions.",
    
    "The development of advanced prosthetics and neural interfaces is restoring mobility and sensory function to individuals with disabilities, while also exploring the potential for human enhancement and brain-computer communication.",
    
    "Precision fermentation technologies are enabling the production of proteins, enzymes, and other biological molecules without traditional agriculture, potentially transforming food production and reducing environmental impact.",
    
    "The integration of artificial intelligence in drug manufacturing is optimizing production processes, ensuring quality control, and enabling more flexible and responsive pharmaceutical supply chains.",
    
    "Advanced recycling technologies are breaking down plastic waste into its molecular components, enabling the creation of new plastics with virgin-like properties and supporting the transition to a circular economy.",
    
    "The field of computational archaeology uses advanced imaging, data analysis, and modeling techniques to uncover new insights about ancient civilizations and preserve cultural heritage for future generations.",
    
    "Bioprinting technologies are advancing toward the goal of printing functional human tissues and organs using living cells, potentially addressing organ shortages and enabling personalized medicine approaches.",
    
    "The development of autonomous ships and maritime robotics is transforming ocean transportation, research, and monitoring by enabling unmanned vessels to operate safely and efficiently in challenging marine environments.",
    
    "Smart grid technologies are enabling bidirectional energy flow, real-time monitoring, and automated response to power outages, supporting the integration of distributed renewable energy sources and electric vehicles.",
    
    "The application of virtual reality in mental health treatment is providing new therapeutic approaches for conditions like PTSD, phobias, and anxiety disorders through controlled exposure therapy and immersive relaxation techniques."
]


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        def monitor():
            while self.monitoring:
                # CPU and Memory
                self.cpu_usage.append(psutil.cpu_percent(interval=1))
                self.memory_usage.append(psutil.virtual_memory().percent)
                
                # GPU monitoring if available
                if GPU_MONITORING:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_load = [gpu.load * 100 for gpu in gpus]
                            gpu_mem = [gpu.memoryUtil * 100 for gpu in gpus]
                            self.gpu_usage.append(gpu_load)
                            self.gpu_memory.append(gpu_mem)
                    except Exception as e:
                        print(f"GPU monitoring error: {e}")
                
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        
        stats = {
            'cpu': {
                'avg': statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                'max': max(self.cpu_usage) if self.cpu_usage else 0,
                'min': min(self.cpu_usage) if self.cpu_usage else 0,
            },
            'memory': {
                'avg': statistics.mean(self.memory_usage) if self.memory_usage else 0,
                'max': max(self.memory_usage) if self.memory_usage else 0,
                'min': min(self.memory_usage) if self.memory_usage else 0,
            }
        }
        
        if self.gpu_usage and GPU_MONITORING:
            # Average across all GPUs and time
            all_gpu_loads = [load for loads in self.gpu_usage for load in loads]
            all_gpu_memory = [mem for mems in self.gpu_memory for mem in mems]
            
            stats['gpu'] = {
                'avg_load': statistics.mean(all_gpu_loads) if all_gpu_loads else 0,
                'max_load': max(all_gpu_loads) if all_gpu_loads else 0,
                'avg_memory': statistics.mean(all_gpu_memory) if all_gpu_memory else 0,
                'max_memory': max(all_gpu_memory) if all_gpu_memory else 0,
            }
        
        return stats


class TTSBenchmark:
    """Comprehensive TTS benchmarking suite."""
    
    def __init__(self):
        self.results = {
            'original': {},
            'optimized': {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
        }
        
        if GPU_MONITORING:
            try:
                gpus = GPUtil.getGPUs()
                info['gpus'] = [
                    {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'driver': gpu.driver
                    }
                    for gpu in gpus
                ]
            except Exception as e:
                info['gpu_error'] = str(e)
        
        return info
    
    def benchmark_original(self, texts: List[str], num_runs: int = 1) -> Dict[str, Any]:
        """Benchmark the original OrpheusCpp implementation."""
        print(f"\n🔥 Benchmarking Original OrpheusCpp ({len(texts)} texts, {num_runs} runs)")
        
        try:
            # Initialize with some GPU acceleration
            model = OrpheusCpp(n_gpu_layers=32, verbose=False)
            
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            times = []
            audio_lengths = []
            errors = 0
            
            start_time = time.time()
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                
                for i, text in enumerate(texts):
                    try:
                        text_start = time.time()
                        sample_rate, audio = model.tts(text)
                        text_end = time.time()
                        
                        duration = text_end - text_start
                        audio_length = len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate
                        
                        times.append(duration)
                        audio_lengths.append(audio_length)
                        
                        if (i + 1) % 10 == 0:
                            print(f"    Completed {i + 1}/{len(texts)} texts")
                            
                    except Exception as e:
                        print(f"    Error processing text {i}: {e}")
                        errors += 1
            
            total_time = time.time() - start_time
            system_stats = monitor.stop_monitoring()
            
            results = {
                'total_time': total_time,
                'avg_time_per_text': statistics.mean(times) if times else 0,
                'median_time_per_text': statistics.median(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'total_audio_length': sum(audio_lengths),
                'avg_audio_length': statistics.mean(audio_lengths) if audio_lengths else 0,
                'texts_per_second': len(texts) * num_runs / total_time if total_time > 0 else 0,
                'real_time_factor': sum(audio_lengths) / sum(times) if sum(times) > 0 else 0,
                'errors': errors,
                'success_rate': (len(texts) * num_runs - errors) / (len(texts) * num_runs) * 100,
                'system_stats': system_stats
            }
            
            print(f"  ✅ Original benchmark completed in {total_time:.2f}s")
            print(f"  📊 Average time per text: {results['avg_time_per_text']:.2f}s")
            print(f"  🎵 Real-time factor: {results['real_time_factor']:.2f}x")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Original benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_optimized(self, texts: List[str], num_runs: int = 1) -> Dict[str, Any]:
        """Benchmark the optimized OrpheusCpp implementation."""
        print(f"\n🚀 Benchmarking Optimized OrpheusCpp ({len(texts)} texts, {num_runs} runs)")
        
        try:
            # Initialize with maximum optimization
            model = OptimizedOrpheusCpp(
                n_gpu_layers=-1,  # All layers on GPU
                n_threads=0,  # Auto-detect
                verbose=False,
                batch_size=16,  # Larger batch size
                n_parallel=8,  # More parallel sessions
            )
            
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
            times = []
            audio_lengths = []
            errors = 0
            
            start_time = time.time()
            
            # Test batch processing
            batch_times = []
            batch_size = 4
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}")
                
                # Process in batches for better GPU utilization
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    try:
                        batch_start = time.time()
                        results = model.batch_tts(batch_texts)
                        batch_end = time.time()
                        
                        batch_duration = batch_end - batch_start
                        batch_times.append(batch_duration)
                        
                        for j, (sample_rate, audio) in enumerate(results):
                            if audio.size > 0:
                                audio_length = len(audio[0]) / sample_rate if len(audio.shape) > 1 else len(audio) / sample_rate
                                audio_lengths.append(audio_length)
                                times.append(batch_duration / len(batch_texts))  # Approximate per-text time
                            else:
                                errors += 1
                        
                        print(f"    Completed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        
                    except Exception as e:
                        print(f"    Error processing batch {i//batch_size}: {e}")
                        errors += len(batch_texts)
            
            total_time = time.time() - start_time
            system_stats = monitor.stop_monitoring()
            
            results = {
                'total_time': total_time,
                'avg_time_per_text': statistics.mean(times) if times else 0,
                'median_time_per_text': statistics.median(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'total_audio_length': sum(audio_lengths),
                'avg_audio_length': statistics.mean(audio_lengths) if audio_lengths else 0,
                'texts_per_second': len(texts) * num_runs / total_time if total_time > 0 else 0,
                'real_time_factor': sum(audio_lengths) / sum(times) if sum(times) > 0 else 0,
                'batch_avg_time': statistics.mean(batch_times) if batch_times else 0,
                'errors': errors,
                'success_rate': (len(texts) * num_runs - errors) / (len(texts) * num_runs) * 100,
                'system_stats': system_stats
            }
            
            print(f"  ✅ Optimized benchmark completed in {total_time:.2f}s")
            print(f"  📊 Average time per text: {results['avg_time_per_text']:.2f}s")
            print(f"  🎵 Real-time factor: {results['real_time_factor']:.2f}x")
            print(f"  🚀 Batch processing avg: {results['batch_avg_time']:.2f}s per batch")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Optimized benchmark failed: {e}")
            return {'error': str(e)}
    
    def run_comparison(self, num_texts: int = 100, num_runs: int = 1) -> Dict[str, Any]:
        """Run comparison between original and optimized versions."""
        print(f"\n🏁 Starting TTS Benchmark Comparison")
        print(f"📝 Testing {num_texts} texts with {num_runs} runs each")
        print(f"💻 System: {self.results['system_info']['cpu_count']} CPU cores, "
              f"{self.results['system_info']['memory_total_gb']:.1f}GB RAM")
        
        if 'gpus' in self.results['system_info']:
            for i, gpu in enumerate(self.results['system_info']['gpus']):
                print(f"🎮 GPU {i}: {gpu['name']} ({gpu['memory_total_mb']}MB)")
        
        # Select subset of texts for testing
        test_texts = BENCHMARK_TEXTS[:num_texts]
        
        # Benchmark original version
        self.results['original'] = self.benchmark_original(test_texts, num_runs)
        
        # Benchmark optimized version
        self.results['optimized'] = self.benchmark_optimized(test_texts, num_runs)
        
        # Calculate improvements
        if ('error' not in self.results['original'] and 
            'error' not in self.results['optimized']):
            
            orig = self.results['original']
            opt = self.results['optimized']
            
            improvements = {
                'speed_improvement': orig['avg_time_per_text'] / opt['avg_time_per_text'] if opt['avg_time_per_text'] > 0 else 0,
                'throughput_improvement': opt['texts_per_second'] / orig['texts_per_second'] if orig['texts_per_second'] > 0 else 0,
                'total_time_reduction': (orig['total_time'] - opt['total_time']) / orig['total_time'] * 100 if orig['total_time'] > 0 else 0,
                'real_time_factor_improvement': opt['real_time_factor'] / orig['real_time_factor'] if orig['real_time_factor'] > 0 else 0,
            }
            
            self.results['improvements'] = improvements
            
            print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
            print(f"  🚀 Speed improvement: {improvements['speed_improvement']:.2f}x faster")
            print(f"  📊 Throughput improvement: {improvements['throughput_improvement']:.2f}x")
            print(f"  ⏱️  Total time reduction: {improvements['total_time_reduction']:.1f}%")
            print(f"  🎵 Real-time factor improvement: {improvements['real_time_factor_improvement']:.2f}x")
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tts_benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print(f"\n" + "="*80)
        print(f"🎯 TTS BENCHMARK SUMMARY")
        print(f"="*80)
        
        # Handle original version results
        if self.results['original'] and 'error' not in self.results['original']:
            orig = self.results['original']
            print(f"\n📊 ORIGINAL VERSION:")
            print(f"  Total time: {orig['total_time']:.2f}s")
            print(f"  Avg time per text: {orig['avg_time_per_text']:.2f}s")
            print(f"  Texts per second: {orig['texts_per_second']:.2f}")
            print(f"  Real-time factor: {orig['real_time_factor']:.2f}x")
            print(f"  Success rate: {orig['success_rate']:.1f}%")
            
            if 'system_stats' in orig and 'gpu' in orig['system_stats']:
                gpu_stats = orig['system_stats']['gpu']
                print(f"  GPU utilization: {gpu_stats['avg_load']:.1f}% avg, {gpu_stats['max_load']:.1f}% max")
                print(f"  GPU memory: {gpu_stats['avg_memory']:.1f}% avg, {gpu_stats['max_memory']:.1f}% max")
        elif self.results['original'] and 'error' in self.results['original']:
            print(f"\n📊 ORIGINAL VERSION:")
            print(f"  ❌ Benchmark failed: {self.results['original']['error']}")
        else:
            print(f"\n📊 ORIGINAL VERSION: Not tested")
        
        # Handle optimized version results
        if self.results['optimized'] and 'error' not in self.results['optimized']:
            opt = self.results['optimized']
            print(f"\n🚀 OPTIMIZED VERSION:")
            print(f"  Total time: {opt['total_time']:.2f}s")
            print(f"  Avg time per text: {opt['avg_time_per_text']:.2f}s")
            print(f"  Texts per second: {opt['texts_per_second']:.2f}")
            print(f"  Real-time factor: {opt['real_time_factor']:.2f}x")
            print(f"  Success rate: {opt['success_rate']:.1f}%")
            
            if 'system_stats' in opt and 'gpu' in opt['system_stats']:
                gpu_stats = opt['system_stats']['gpu']
                print(f"  GPU utilization: {gpu_stats['avg_load']:.1f}% avg, {gpu_stats['max_load']:.1f}% max")
                print(f"  GPU memory: {gpu_stats['avg_memory']:.1f}% avg, {gpu_stats['max_memory']:.1f}% max")
        elif self.results['optimized'] and 'error' in self.results['optimized']:
            print(f"\n🚀 OPTIMIZED VERSION:")
            print(f"  ❌ Benchmark failed: {self.results['optimized']['error']}")
        else:
            print(f"\n🚀 OPTIMIZED VERSION: Not tested")
        
        # Handle improvements calculation
        if ('improvements' in self.results and 
            self.results['original'] and 'error' not in self.results['original'] and
            self.results['optimized'] and 'error' not in self.results['optimized']):
            imp = self.results['improvements']
            print(f"\n📈 PERFORMANCE GAINS:")
            print(f"  Speed improvement: {imp['speed_improvement']:.2f}x")
            print(f"  Throughput improvement: {imp['throughput_improvement']:.2f}x")
            print(f"  Time reduction: {imp['total_time_reduction']:.1f}%")
            print(f"  Real-time factor improvement: {imp['real_time_factor_improvement']:.2f}x")
        else:
            print(f"\n📈 PERFORMANCE GAINS: Cannot calculate (one or both benchmarks failed)")
        
        print(f"\n" + "="*80)


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="TTS Performance Benchmark")
    parser.add_argument("--texts", type=int, default=100, 
                       help="Number of texts to benchmark (default: 100)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per benchmark (default: 1)")
    parser.add_argument("--original-only", action="store_true",
                       help="Only benchmark original version")
    parser.add_argument("--optimized-only", action="store_true",
                       help="Only benchmark optimized version")
    parser.add_argument("--output", type=str,
                       help="Output filename for results JSON")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.texts > len(BENCHMARK_TEXTS):
        print(f"Warning: Only {len(BENCHMARK_TEXTS)} texts available, using all of them")
        args.texts = len(BENCHMARK_TEXTS)
    
    benchmark = TTSBenchmark()
    
    try:
        if args.original_only:
            test_texts = BENCHMARK_TEXTS[:args.texts]
            benchmark.results['original'] = benchmark.benchmark_original(test_texts, args.runs)
        elif args.optimized_only:
            test_texts = BENCHMARK_TEXTS[:args.texts]
            benchmark.results['optimized'] = benchmark.benchmark_optimized(test_texts, args.runs)
        else:
            benchmark.run_comparison(args.texts, args.runs)
        
        benchmark.print_summary()
        filename = benchmark.save_results(args.output)
        
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📄 Detailed results saved to: {filename}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
