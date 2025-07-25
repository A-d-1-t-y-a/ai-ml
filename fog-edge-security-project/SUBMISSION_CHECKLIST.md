# Fog and Edge Computing Project: Submission Checklist

## Required Deliverables

### 1. Source Code
- [x] Complete Java Maven project with all source files
- [x] Fully commented code
- [x] pom.xml with all dependencies

### 2. Documentation
- [x] README.md with project overview and setup instructions
- [x] ARCHITECTURE_DIAGRAM.md with system architecture and data flow
- [x] EVALUATION.md with detailed analysis of the prototype
- [x] PROJECT_REPORT.md (5-8 pages) covering implementation and findings
- [x] VIDEO_PRESENTATION_SCRIPT.md for the 5-minute presentation

### 3. Video Presentation
- [ ] 5-minute video presentation demonstrating the prototype
  - Record using screen recording software (OBS, Zoom, etc.)
  - Follow the script in VIDEO_PRESENTATION_SCRIPT.md
  - Upload to a sharing platform or include in the submission package

## Packaging Instructions

1. **Compile the Project**
   ```bash
   mvn clean package
   ```

2. **Create a ZIP Archive**
   - Include all source code
   - Include all documentation files
   - Include the compiled JAR file
   - Include the video presentation (or link to it)

3. **Final Directory Structure for Submission**
   ```
   fog-edge-security-project/
   ├── src/                           # Source code
   ├── target/                        # Compiled code
   │   └── fog-edge-security-project-1.0-SNAPSHOT-jar-with-dependencies.jar
   ├── README.md                      # Project overview
   ├── ARCHITECTURE_DIAGRAM.md        # System architecture
   ├── EVALUATION.md                  # Prototype evaluation
   ├── PROJECT_REPORT.md              # Detailed project report
   ├── VIDEO_PRESENTATION_SCRIPT.md   # Script for video presentation
   ├── SUBMISSION_CHECKLIST.md        # This file
   ├── pom.xml                        # Maven configuration
   └── presentation.mp4               # Video presentation (or link)
   ```

## Submission Steps

1. **Create the Video Presentation**
   - Set up a slide deck based on the script
   - Record your screen and voice
   - Keep the presentation under 5 minutes
   - Save in MP4 format

2. **Compile the Project**
   - Run `mvn clean package` to create the JAR file
   - Verify the JAR file is created in the target directory

3. **Create a ZIP Archive**
   - Zip the entire project folder
   - Name it `fog-edge-security-project_submission.zip`

4. **Submit the ZIP Archive**
   - Upload to the submission portal
   - Include any required submission comments

## Pre-Submission Verification

- [ ] All code compiles without errors
- [ ] All documentation is complete and accurate
- [ ] Video presentation is clear and under 5 minutes
- [ ] All files are included in the ZIP archive
- [ ] ZIP archive is under the size limit for submission

## Notes

- The project is due on 28/07/2025
- Make sure to submit at least 24 hours before the deadline to avoid technical issues
- Keep a backup copy of your submission
