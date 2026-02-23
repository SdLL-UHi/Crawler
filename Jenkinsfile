@Library('web-service-helper-lib') _

pipeline {
    agent { label 'docker' }

    options {
        ansiColor('xterm')
    }

    stages {
        stage('Build Docker Image') {
            agent {
                docker {
                    image 'docker:24'
                    reuseNode true
                }
            }
            environment {
                HOME = "${env.WORKSPACE}"
                DOCKER_CONFIG = "${env.WORKSPACE}/.docker"
            }
            steps {
                sh 'mkdir -p "$DOCKER_CONFIG"'
                script {
                    ssedocker {
                        create { target "ghcr.io/sdll-uhi/crawler:${env.BUILD_NUMBER}" }
                        publish { tag 'latest' }
                    }
                }
            }
            post {
                always {
                    sh 'rm -rf "$DOCKER_CONFIG" || true'
                }
            }
        }
    }
}