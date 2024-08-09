pipeline {
    agent any

    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        registry = 'dixuson/controller_bot'
        dockerCredential = 'dockerhub'      
    }

    stages {
        stage('Docker Build') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    sh 'docker build -t dixuson/controller_bot --load --rm -f ./dockerfiles/Dockerfile_controller .'
                    // dockerImage = docker.build registry + ":$BUILD_NUMBER" 
                    // echo 'Pushing image to dockerhub..'
                    // docker.withRegistry( '', registryCredential ) {
                    //     dockerImage.push()
                    //     dockerImage.push('latest')
                    // }
                }
            }
        }
        stage('Docker Push') {
            agent any
            steps {
                withCredentials([usernamePassword(credentialsId: ${env.dockerCredential}, passwordVariable: 'dockerHubPassword', usernameVariable: 'dockerHubUser')]) {
                    sh "docker login -u ${env.dockerHubUser} -p ${env.dockerHubPassword}"
                    sh "docker push ${env.registry}:latest"
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
