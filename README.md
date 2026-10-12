AI-Based Photo Segregation Using Merge Sort
An AI-powered photo organization system that detects faces in event photographs, groups photos by person, and organizes Person–Photo records using a custom recursive Merge Sort algorithm.
The project combines pretrained face recognition, inference-time filtering, face matching, clustering, and Design and Analysis of Algorithms (DAA).

1. How to Run the Project
The application consists of two development servers:
* Backend: Python + FastAPI
* Frontend: React + Vite
Start both servers in separate terminal windows.
Prerequisites
* Python 3.13 or a compatible Python environment
* Node.js and npm
* Git
* The project's existing Python virtual environment and dependencies
* Internet access may be required the first time the pretrained InsightFace model is downloaded.
Step 1 — Open the Project Directory
Open Terminal and navigate to the project root:
git clone https://github.com/adityaagurav/PhotoSeggregation.git
cd PhotoSeggregation
If you cloned the repository somewhere else, replace the path with your own project directory.
Step 2 — Start the Python API Server
In the first terminal, run:
.venv/bin/python run_api.py
The FastAPI backend should be available at:
* API base URL: http://127.0.0.1:8000
* Health endpoint: http://127.0.0.1:8000/health
* API documentation: http://127.0.0.1:8000/docs
Keep this terminal running.
The backend handles photo uploads, invokes the existing face-processing pipeline, tracks processing jobs, and supplies the results to the frontend.
Step 3 — Start the React Frontend
Open a second terminal and navigate to the project:
cd "/Users/adityagurav/Documents/DAA LAB/photo-seggregation_FA2"
cd frontend
npm run dev
Vite will display the local development URL, normally:
http://localhost:5173
Keep this terminal running as well.
Step 4 — Use the Application
1. Open http://localhost:5173 in your browser.
2. Navigate to the upload page.
3. Select or drag and drop your event photographs.
4. Click Process to run the face-processing pipeline.
5. Wait for processing to finish.
6. Open the results page to explore the identified people and their associated photographs.
7. Click different Person folders to inspect the photos assigned to each person.
The application displays the processing statistics, person counts, photo associations, and Merge Sort verification status.
Important: Start the backend before processing photos. If either server is stopped, restart it in its respective terminal.
Running the Original Pipeline Separately
To run the original command-line pipeline without the web interface, use the project root:
.venv/bin/python run.py
This uses the pipeline's existing default output behavior.

2. Project Overview
Problem Statement
Large collections of event photographs often contain the same individuals across multiple images. Manually locating and organizing all photographs belonging to a particular person is time-consuming.
This project automates the process by detecting faces, generating facial embeddings, matching faces across photographs, grouping them into person identities, and organizing the resulting Person–Photo records.
The system assigns anonymous identifiers such as Person #1, Person #2, and Person #3, rather than identifying individuals by name.
Objectives
* Detect faces in uploaded event photographs.
* Reduce false-positive and duplicate face detections.
* Generate facial embeddings using a pretrained recognition model.
* Match and cluster faces across different photographs.
* Handle uncertain detections through cross-image matching.
* Create Person–Photo records.
* Sort records using a custom recursive Merge Sort implementation.
* Display organized results through a web interface.
* Preserve original photographs during processing.

3. Technology Stack
Component	Technology
Programming language	Python
Face detection and embeddings	InsightFace
Pretrained model	buffalo_l
Image processing	OpenCV
Numerical operations	NumPy
Backend API	FastAPI
API server	Uvicorn
Frontend	React
Frontend tooling	Vite
DAA algorithm	Custom recursive Merge Sort
Testing	Python unittest
Data and reports	JSON and CSV
The project uses pretrained inference rather than fine-tuning the face recognition model on a custom dataset.

4. System Architecture
The application follows this processing flow:
User uploads photographs
          |
          v
    FastAPI Backend
          |
          v
     Face Detection
          |
          v
 Inference-Time Filtering
          |
          v
    Face Embeddings
          |
          v
 Cross-Image Face Matching
          |
          v
       Clustering
          |
          v
 Anonymous Person IDs
          |
          v
    Person–Photo Records
          |
          v
   Custom Merge Sort
          |
          v
    Organized Results
          |
          v
    React Frontend
The frontend communicates with the backend through API endpoints. The backend invokes the existing processing pipeline and exposes job status and results.

5. Inference-Time Filtering
5.1 Overview
The project uses the pretrained InsightFace buffalo_l model. The model is not fine-tuned or retrained as part of this project.
Instead, the detection pipeline applies additional inference-time filtering to improve the quality of detected face regions and reduce unwanted background detections.
The purpose is to retain useful faces while reducing false positives without unnecessarily discarding legitimate people.
5.2 Face-Size Filtering
Faces with very small bounding boxes can represent distant background individuals or unreliable detections.
The existing size-based filtering logic evaluates detected face regions and rejects detections that do not meet the established criteria.
Size filtering is not sufficient on its own: a background face can occasionally be slightly larger than the minimum size threshold.
5.3 Perspective-Aware Relative-Scale Filtering
A perspective-aware relative-scale signal helps identify suspiciously small faces relative to other faces in the same photograph.
This additional signal complements the existing size filter. It is intended to reduce false detections of background faces without indiscriminately rejecting all small faces.
5.4 Face-Quality Signals
The existing pipeline also uses face-quality signals such as:
* Blur: helps assess whether a detected face is sufficiently clear.
* Yaw: provides a signal about the face's orientation.
These signals are used within the existing filtering logic. They should not be interpreted as independent proof that a detection is incorrect.
5.5 Detector Resolution
The detector supports the configurable --det-size option.
The default detection size is:
--det-size 640
Testing at a larger detection size, such as 1280, can expose additional small faces. However, increasing detection resolution may also introduce extra background detections, so the existing filtering safeguards remain important.
The default setting and established filtering thresholds are preserved.

6. Duplicate Face-Detection Fix
6.1 Problem
A single physical face may occasionally be represented by more than one overlapping detection box. If duplicate detections continue through the pipeline, they can inflate face counts and potentially affect later processing.
The project now includes a dedicated bounding-box deduplication safeguard.
6.2 Implemented Functions
The duplicate-detection logic in ml/inference/predict.py includes:
* bbox_iou
* bbox_centre_distance
* bbox_mean_side
* duplicate_box_reason
* deduplicate_faces
These functions help assess overlap, spatial proximity, and the relative geometry of detected bounding boxes.
Deduplication is applied during face-region detection. The pipeline also records how many duplicate detections were removed.
6.3 Why Both Overlap and Distance Matter
Intersection over Union (IoU) measures the overlap between two bounding boxes:
[ IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}} ]
A high overlap can indicate duplicate detections, but overlap alone does not prove that two boxes represent the same person.
The implementation therefore also considers center distance and bounding-box size safeguards. This helps avoid incorrectly merging nearby, distinct faces or discarding legitimate smaller faces.
6.4 Verification Results
The regression tests include cases for:
1. Highly overlapping boxes representing the same face.
2. Nearby but distinct faces.
3. Legitimate small faces.
4. Duplicate processing and counting.
5. Correct Person–Photo records and Merge Sort verification.
The six-person test photograph, group_faces.jpg, was tested with duplicate detection responses deliberately re-injected.
Measurement	Result
Boxes before deduplication	8
Boxes after deduplication	6
Legitimate faces preserved	6
The six surviving boxes corresponded to the detector's original six boxes.
Important limitation: The bundled real photographs did not reproduce duplicate boxes from the detector during the reported test run. The normal dataset therefore showed zero duplicate detections suppressed. The 8-to-6 result verifies the deduplication logic under a controlled reproduction; it does not establish that the real detector routinely generates duplicate boxes.

7. Face Embeddings and Matching
After filtering, the pipeline generates facial embeddings for eligible face detections.
The pretrained InsightFace model produces 512-dimensional face embeddings, which are used for face comparison and matching.
The existing pipeline distinguishes between:
* KEEP: a face accepted by the filtering stage.
* UNCERTAIN: a face that requires further consideration.
* FILTERED: a face rejected by the filtering stage.
KEEP and UNCERTAIN faces may receive embeddings. FILTERED faces must never receive embeddings.
Cross-Image Uncertain-Face Recovery
An uncertain face may be promoted to CONFIRMED_FROM_MATCH only when its strongest eligible match from another image reaches the configured similarity threshold.
The existing default/configuration used in this workflow is a similarity threshold of 0.70.
The recovery logic preserves the original uncertainty information and records matching metadata, including the matched photo/face, similarity, threshold, and final status in:
data/output/confirmed_uncertain_matches.csv
The similarity threshold is a configurable matching criterion, not a guarantee that every match is correct.

8. Clustering and Person IDs
Once eligible face embeddings and cross-image matches are available, the existing clustering stage groups faces that belong to the same person.
The system assigns anonymous Person IDs dynamically.
For example:
Person #1
  ├── event01.jpeg
  ├── event04.jpeg
  └── event07.jpeg

Person #2
  ├── event02.jpeg
  └── event07.jpeg
A photograph containing multiple identified people can be associated with multiple Person IDs.
These associations are intentional: the same original photograph can appear in multiple person's organized collections without being modified.

9. Merge Sort — DAA Component
9.1 Purpose
Merge Sort is the explicit Data Structures and Algorithms component of this project.
After the face-processing pipeline creates Person–Photo records, the project uses a custom recursive Merge Sort implementation to organize records by Person ID.
9.2 Algorithm
The implementation follows the divide-and-conquer approach:
1. Divide the record list into two halves.
2. Recursively sort each half.
3. Merge the sorted halves by comparing their Person ID values.
4. Continue until the complete record list is sorted.
The project uses its own recursive implementation rather than relying on Python's built-in sorted() or list.sort() for the required sorting operation.
9.3 Complexity Analysis
Case	Time Complexity
Best case	(O(n\log n))
Average case	(O(n\log n))
Worst case	(O(n\log n))
Auxiliary space	(O(n))
Here, (n) is the number of Person–Photo records being sorted.
Merge Sort provides predictable (O(n\log n)) time complexity, making it suitable for organizing larger record collections.
9.4 Verification
The pipeline verifies the resulting record order and reports whether Merge Sort passed validation.
A PASS status must reflect actual verification rather than simply being hardcoded into the interface.
The frontend displays the backend's reported Merge Sort status and associated record count.

10. Backend API
The FastAPI application connects the frontend to the processing pipeline.
Endpoint	Purpose
GET /health	Check backend health
POST /upload	Upload and validate photographs
POST /process/{job_id}	Start processing for a job
GET /status/{job_id}	Retrieve job-processing status
GET /results/{job_id}	Retrieve processing results and statistics
GET /photos/{job_id}/{filename}	Serve an original uploaded photograph through a job-scoped endpoint
The backend isolates job uploads and outputs and validates requests to prevent unsafe file access.
The results endpoint provides information such as person associations, record counts, sorting verification, and duplicate-detection statistics.
The job registry is in memory, while supported result data can be rebuilt from persisted outputs. Transient metrics that are not persisted may not be available after a backend restart.
For full request and response details, open:
http://127.0.0.1:8000/docs

11. Frontend Features
The React frontend provides a Finder-inspired dark interface for viewing the results.
Current features include:
* Multiple-photo upload and processing.
* A person-wise folder grid.
* Dynamic Person IDs and photo counts.
* A detail gallery for the selected person.
* Original filenames and image thumbnails.
* Shared-photo associations across multiple people.
* Search and grid/list views.
* Processing status and error states.
* Person, photo, and record statistics.
* Merge Sort PASS/FAIL verification.
* An uncertain-face metadata view.
* Refresh and result navigation.
The folder-style interface currently represents organized results in the UI. Physical person-wise output directories and ZIP export are separate features and should not be considered implemented until they are added and tested.

12. Output Files
The pipeline uses JSON and CSV files to persist processing information.
File	Purpose
person_clusters.json	Person IDs and their associated photos/faces
person_photo_records.csv	Person–Photo records, including sorted output
confirmed_uncertain_matches.csv	Uncertain-face recovery and match metadata
singleton_validation.csv	Singleton validation information
Typical output locations include:
data/
├── uploads/
└── output/
The API can maintain job-specific data in subdirectories under these locations.
Exact generated files depend on the processing path and job.

13. Testing and Verification
The reported regression-test command is:
.venv/bin/python -m unittest discover -s tests
The duplicate-detection fix was verified with:
* 18 dedicated duplicate-detection tests.
* 91 passing tests across the complete unittest suite.
* Preservation checks for face embeddings and filtering statuses.
* A controlled eight-box-to-six-box duplicate-detection test.
* A real-dataset run that retained the existing person associations and sorting behavior.
The reported real-dataset run produced:
Metric	Result
Raw face detections	43
Duplicate detections suppressed	0
Retained faces	33
Identified persons	13
Person–Photo records	28
Merge Sort	PASS
These figures describe the tested dataset and are not guaranteed for every upload.
The following commands verify the frontend:
cd frontend
npm run lint
npm run build
The reported test run completed both frontend checks successfully.

14. Known Limitations
* Face detection and filtering depend on image quality, face size, pose, occlusion, and detector behavior.
* Inference-time filtering reduces unwanted detections but cannot guarantee perfect detection.
* Bounding-box deduplication is a safeguard; its real-world effect depends on whether duplicate detections occur.
* Cross-image matching depends on embedding quality and the configured similarity threshold.
* A correct face count does not, by itself, guarantee correct identity clustering.
* Some transient processing statistics may not survive a backend restart.
* Physical person-wise folders, ZIP export, and face previews should only be described as available after their respective implementations and tests are completed.

15. Development Principles
The following constraints are important to the project:
* Use pretrained InsightFace inference; do not describe inference-time filtering as model fine-tuning.
* Preserve the established filtering thresholds and detector defaults unless an intentional change is required and verified.
* Do not generate embeddings for FILTERED faces.
* Preserve the existing uncertain-face recovery behavior.
* Keep Person IDs consistent with the actual clustering results.
* Use the custom recursive Merge Sort for the required sorting operation.
* Preserve original uploaded photographs.
* Validate API inputs and prevent path traversal.
* Document test results honestly, including limitations and cases that could not be reproduced.

16. Project Summary
AI-Based Photo Segregation Using Merge Sort demonstrates how computer vision and algorithm design can work together to automate photo organization.
The AI pipeline detects faces, filters unreliable detections, creates embeddings, matches faces across images, and groups them into anonymous person identities. A custom recursive Merge Sort then organizes the resulting Person–Photo records.
The FastAPI backend exposes processing functionality and results, while the React frontend provides an interactive interface for exploring the organized photographs.
The project combines practical AI inference, defensive detection filtering, API development, frontend engineering, testing, and algorithmic complexity analysis in a single application.
