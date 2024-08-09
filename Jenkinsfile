pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        dockerCredential = 'dockerhub'
        imageTag_controller = 'dixuson/controller_bot'
        dockerfile_controller = './dockerfiles/Dockerfile_controller' 
        imageTag_app = 'dixuson/app_bot'
        dockerfile_app = './dockerfiles/Dockerfile_app'      
        imageTag_app_knowledge = 'dixuson/app_knowledge_bot'
        dockerfile_app_knowledge = './dockerfiles/Dockerfile_app_knowledge'
        imageTag_retrieval_worker = 'dixuson/retrieval_worker_bot'
        dockerfile_retrieval_worker = './dockerfiles/Dockerfile_retrieval_worker'
    }

    stages {
        stage('Docker Build') {
            steps {
                script {
                    echo 'Building image controller bot for deployment..'
                    // sh 'docker build -t dixuson/controller_bot --load --rm -f ./dockerfiles/Dockerfile_controller .'
                    dockerImage_controller = docker.build(imageTag_controller, "-f ${env.dockerfile_controller} .") 
                    echo 'Building image app bot for deployment..'
                    dockerImage_app = docker.build(imageTag_app, "-f ${env.dockerfile_app} .") 
                    echo 'Building image app knowledge bot for deployment..'
                    dockerImage_app_knowledge = docker.build(imageTag_app_knowledge, "-f ${env.dockerfile_app_knowledge} .") 
                    echo 'Building image retrieval worker bot for deployment..'
                    dockerImage_retrieval_worker = docker.build(imageTag_retrieval_worker, "-f ${env.dockerfile_retrieval_worker} .") 
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
                        dockerImage_controller.push('latest')
                        dockerImage_app.push('latest')
                        dockerImage_app_knowledge.push('latest')
                        dockerImage_retrieval_worker.push('latest')
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
