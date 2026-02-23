@Library('web-service-helper-lib') _

pipeline {
    agent { label 'docker' }

    options {
        ansiColor('xterm')
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    ssedocker {
                        create { target "ghcr.io/sdll-uhi/crawler:${env.BUILD_NUMBER}" }
                        publish { tag latest }
                    }
                }
            }
        }
    }
}