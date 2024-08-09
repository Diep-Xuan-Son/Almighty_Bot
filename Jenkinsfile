pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        imageTag = 'dixuson/controller_bot'
        dockerCredential = 'dockerhub'
        dockerfile = './dockerfiles/Dockerfile_controller'      
    }

    stages {
        stage('Docker Build') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    // sh 'docker build -t dixuson/controller_bot --load --rm -f ./dockerfiles/Dockerfile_controller .'
                    dockerImage = docker.build(imageTag, "-f ${env.dockerfile} .")              

                }
            }
        }
        stage('Docker Push') {
            agent any
            steps {
                // withCredentials([usernamePassword(credentialsId: dockerCredential, passwordVariable: 'dockerHubPassword', usernameVariable: 'dockerHubUser')]) {
                //    sh "docker login -u ${env.dockerHubUser} -p ${env.dockerHubPassword}"
                //    sh "docker push ${env.imageTag}:latest"
                // }
                script {
                    echo 'Pushing image to dockerhub...'
                    docker.withRegistry( '', dockerCredential ) {
                        // sh "docker push ${env.imageTag}:latest"
                        dockerImage.push('latest')
                    }
                }
            }
        }
        // stage('Test') {
        //     agent any
        //     steps {
        //         echo 'Testing model correctness..'
        //         sh "curl -X 'POST' 'http://192.168.6.142:21001/worker_get_status' -H 'accept: -H 'accept: application/json'"
        //     }
        // }
        // stage('Deploy') {
        //     steps {
        //         echo 'Deploying models..'
        //         echo 'Running a script to trigger pull and start a docker container'
        //     }
        // }
    }
}

// token_almighty_bot: dckr_pat_shd5j1x2@@@@@@@@@@@@S09VDQQdTmcN-ZvR_IA
