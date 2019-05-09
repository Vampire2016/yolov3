export default {
    namespace: 'onnx',

    state: {
        name: 'lonton'
    },

    effects: {
        *getModel(action, { call, put }) {
            const getArtifactRequest = new Request(require(`../mlmodel/${action.payload.modelName}.onnx`), {
                method: 'GET',
                // headers: new Headers(getRequestHeaders(document.cookie))
            })
            return fetch(getArtifactRequest)
        },
    },

    reducers: {
        
    }
}